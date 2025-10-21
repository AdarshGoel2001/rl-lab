"""Orchestrator coordinating world-model workflows."""

from __future__ import annotations

import logging
import copy
import time
from collections import deque
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from ..data_sources import DataSource, create_data_source
from .factory import ComponentFactory
from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config, resolve_device, save_config
from ..utils.logger import create_logger
from ..utils.registry import auto_import_modules, get_buffer, get_environment
from ..workflows.world_models.base import CollectResult, WorldModelWorkflow
from ..workflows.world_models.context import WorkflowContext, WorldModelComponents
from ..workflows.world_models.controllers import ControllerManager
from .phase_scheduler import PhaseScheduler, PhaseDefinition

logger = logging.getLogger(__name__)


class WorldModelOrchestrator:
    """Owns experiment lifecycle while delegating algorithm detail to workflows."""

    def __init__(
        self,
        config: Config,
        workflow: WorldModelWorkflow,
        *,
        experiment_dir: Optional[Path] = None,
        scheduler: Optional["PhaseScheduler"] = None,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.device = resolve_device(config.experiment.device)
        self.scheduler = scheduler or self._build_scheduler()
        self.data_sources: Dict[str, DataSource] = {}
        self.controllers: Dict[str, Any] = {}
        self.controller_manager: Optional[ControllerManager] = None
        self.global_step = 0

        timestamp = config.experiment.timestamp if hasattr(config.experiment, "timestamp") else None
        if timestamp is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        default_dir = Path("experiments") / f"{config.experiment.name}_{timestamp}"
        self.experiment_dir = Path(experiment_dir) if experiment_dir else default_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        auto_import_modules()

        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
            auto_save_frequency=self.config.training.checkpoint_frequency,
            max_checkpoints=5,
        )
        self.experiment_logger = create_logger(
            self.experiment_dir,
            self.config.logging.__dict__,
            self.config.to_dict(),
        )

        self._context: Optional[WorkflowContext] = None
        self._initialized = False

    def ensure_context(self) -> WorkflowContext:
        if self._context is None:
            self._context = self._build_context()
            self.global_step = self._context.global_step
        return self._context

    def _build_scheduler(self) -> PhaseScheduler:
        raw_phases = getattr(self.config.training, "phases", None)
        phases_config: Optional[list[Dict[str, Any]]] = None

        if raw_phases:
            phases_config = []
            for entry in raw_phases:
                if hasattr(entry, "to_dict"):
                    phases_config.append(entry.to_dict())
                elif isinstance(entry, dict):
                    phases_config.append(copy.deepcopy(entry))
                else:
                    phases_config.append(copy.deepcopy(getattr(entry, "__dict__", {})))

        return PhaseScheduler(phases_config)

    def _build_context(self) -> WorkflowContext:
        logger.info("Building workflow context for orchestrator path")
        env_cfg = copy.deepcopy(self.config.environment.__dict__)
        environment_cls = get_environment(self.config.environment.wrapper)
        train_environment = environment_cls(env_cfg)

        if getattr(self.config, "evaluation", None):
            eval_cfg = copy.deepcopy(self.config.evaluation.__dict__)
            eval_cls = get_environment(self.config.evaluation.wrapper)
            eval_environment = eval_cls(eval_cfg)
        else:
            eval_cfg = copy.deepcopy(self.config.environment.__dict__)
            if eval_cfg.get("wrapper") == "vectorized_gym":
                eval_cfg["wrapper"] = "gym"
                eval_cfg.pop("num_envs", None)
                eval_cfg.pop("vectorization", None)
            eval_cls = get_environment(eval_cfg["wrapper"])
            eval_environment = eval_cls(eval_cfg)

        component_spec = self._prepare_component_spec(train_environment)
        raw_components = ComponentFactory.create_world_model_components(
            component_spec,
            device=self.device,
            paradigm_config=component_spec.get("paradigm_config", {}),
        )
        raw_components.to(self.device)
        components = WorldModelComponents(**raw_components.as_dict())

        buffer_cls = get_buffer(self.config.buffer.type)
        buffer_cfg = dict(self.config.buffer.__dict__)
        buffer_cfg["device"] = self.device
        buffer_cfg.setdefault("num_envs", getattr(train_environment, "num_envs", 1))
        buffer = buffer_cls(buffer_cfg)

        context = WorkflowContext(
            config=self.config,
            device=self.device,
            train_environment=train_environment,
            eval_environment=eval_environment,
            components=components,
            buffer=buffer,
            checkpoint_manager=self.checkpoint_manager,
            experiment_logger=self.experiment_logger,
        )

        controllers_cfg = getattr(self.config, "controllers", None) or {}
        controllers_cfg = self._prepare_controller_config(controllers_cfg, components)
        controllers = ComponentFactory.create_controllers(
            controllers_cfg,
            device=self.device,
            components=components,
        )
        controller_manager = ControllerManager(controllers)
        context = context.with_updates(
            controllers=controllers or None,
            controller_manager=controller_manager,
        )
        self.controllers = controllers
        self.controller_manager = controller_manager

        data_sources = self._initialize_data_sources(context, buffer)
        context = context.with_updates(data_sources=data_sources)
        self.data_sources = data_sources

        return context

    def _initialize_data_sources(
        self,
        context: WorkflowContext,
        buffer: Any,
    ) -> Dict[str, DataSource]:
        configured_sources = getattr(self.config, "data_sources", None) or {}
        sources: Dict[str, DataSource] = {}

        if not configured_sources:
            default = create_data_source("replay", config={"buffer": buffer})
            default.initialize(context)
            sources["replay"] = default
            return sources

        for name, entry in configured_sources.items():
            if entry is None:
                continue
            normalized = self._normalize_source_entry(entry)
            source_type = str(normalized.get("type") or name or "replay")
            raw_config = normalized.get("config") or {}
            if hasattr(raw_config, "to_dict"):
                raw_config = raw_config.to_dict()
            elif isinstance(raw_config, Mapping):
                raw_config = copy.deepcopy(dict(raw_config))
            else:
                raw_config = {}

            if source_type in {"replay", "world_model_replay"} and "buffer" not in raw_config and "buffer_type" not in raw_config:
                raw_config["buffer"] = buffer

            source = create_data_source(source_type, config=raw_config)
            source.initialize(context)
            sources[name] = source

        if "replay" not in sources:
            fallback = create_data_source("replay", config={"buffer": buffer})
            fallback.initialize(context)
            sources["replay"] = fallback

        return sources

    def _normalize_source_entry(self, entry: Any) -> Dict[str, Any]:
        if isinstance(entry, str):
            return {"type": entry}
        if isinstance(entry, Mapping):
            return copy.deepcopy(dict(entry))
        if hasattr(entry, "to_dict"):
            return entry.to_dict()
        raise TypeError(f"Unsupported data source configuration type: {type(entry)}")

    def _prepare_component_spec(self, environment) -> Dict[str, Any]:
        obs_space = environment.observation_space
        action_space = environment.action_space

        component_config = copy.deepcopy(self.config.components)
        component_spec = dict(component_config)
        component_spec.setdefault("paradigm_config", {})
        component_spec["paradigm_config"] = dict(component_spec["paradigm_config"])
        component_spec["paradigm_config"].setdefault("device", self.device)

        def _inject(key: str) -> Dict:
            entry = component_spec.get(key, {})
            entry = dict(entry)
            entry.setdefault("config", {})
            entry["config"] = dict(entry["config"])
            entry["config"].setdefault("device", self.device)
            component_spec[key] = entry
            return entry["config"]

        encoder_cfg = _inject("encoder")
        if hasattr(obs_space, "shape"):
            shape = obs_space.shape
            if getattr(environment, "is_vectorized", False) and shape:
                shape = tuple(shape[1:]) if len(shape) > 1 else shape
            encoder_cfg.setdefault("input_dim", shape)

        if "policy_head" in component_spec:
            policy_cfg = _inject("policy_head")
            if getattr(action_space, "discrete", False):
                policy_cfg.setdefault("action_dim", action_space.n)
                policy_cfg.setdefault("discrete_actions", True)
            else:
                action_dim = getattr(action_space, "shape", None) or (0,)
                if getattr(environment, "is_vectorized", False) and action_dim:
                    action_dim = action_dim[1:] if len(action_dim) > 1 else action_dim
                policy_cfg.setdefault("action_dim", int(np.prod(action_dim)))
                policy_cfg.setdefault("discrete_actions", False)

        _inject("representation_learner")
        dynamics_cfg = _inject("dynamics_model")
        if getattr(action_space, "discrete", False):
            dynamics_cfg.setdefault("action_dim", int(action_space.n))
            dynamics_cfg.setdefault("discrete_actions", True)
        else:
            action_dim = getattr(action_space, "shape", None) or (0,)
            if getattr(environment, "is_vectorized", False) and action_dim:
                action_dim = action_dim[1:] if len(action_dim) > 1 else action_dim
            flat_dim = int(np.prod(action_dim))
            dynamics_cfg.setdefault("action_dim", flat_dim)
            dynamics_cfg.setdefault("discrete_actions", False)
        if "value_function" in component_spec:
            _inject("value_function")
        _inject("reward_predictor")
        _inject("observation_decoder")
        if "planner" in component_spec:
            _inject("planner")

        paradigm_overrides = getattr(self.config, "paradigm_config", {}) or {}
        if paradigm_overrides:
            component_spec["paradigm_config"].update(paradigm_overrides)

        return component_spec
    def _prepare_controller_config(
        self,
        controller_cfg: Mapping[str, Any],
        components: WorldModelComponents,
    ) -> Dict[str, Any]:
        """Inject default latent/action specs into controller configuration."""
        normalised: Dict[str, Any] = {}
        specs = getattr(components, "specs", {}) or {}
        representation_dim = specs.get("representation_dim")
        action_dim = specs.get("action_dim")
        discrete_actions = specs.get("discrete_actions")

        for role, entry in (controller_cfg or {}).items():
            if entry is None:
                continue
            if hasattr(entry, "to_dict"):
                payload = entry.to_dict()
            elif isinstance(entry, Mapping):
                payload = copy.deepcopy(dict(entry))
            else:
                raise TypeError(
                    f"Controller configuration for role '{role}' must be dict-like, got {type(entry)}"
                )

            config_block = payload.setdefault("config", {})
            if representation_dim is not None:
                config_block.setdefault("representation_dim", representation_dim)
            if action_dim is not None:
                config_block.setdefault("action_dim", action_dim)
            if discrete_actions is not None:
                config_block.setdefault("discrete_actions", bool(discrete_actions))

            normalised[role] = payload

        return normalised

    def initialize(self) -> None:
        if self._initialized:
            return
        context = self.ensure_context()
        self.workflow.initialize(context)
        self._initialized = True
        self.global_step = context.global_step

    def run(self) -> Dict[str, float]:
        self.initialize()
        logger.info("Starting orchestrated world-model training loop")

        total_steps = int(self.config.training.total_timesteps)
        log_freq = int(getattr(self.config.logging, "log_frequency", 0) or 0)
        eval_freq = int(getattr(self.config.training, "eval_frequency", 0) or 0)
        ckpt_freq = int(getattr(self.config.training, "checkpoint_frequency", 0) or 0)

        last_log = self.global_step
        last_eval = self.global_step
        last_ckpt = self.global_step

        start_time = time.time()
        last_update_metrics: Dict[str, float] = {}
        last_batch: Optional[Any] = None

        scheduler = self.scheduler
        if scheduler is None:
            scheduler = self._build_scheduler()
            self.scheduler = scheduler

        while (
            self.global_step < total_steps
            and not scheduler.is_finished()
        ):
            phase_def = scheduler.current_phase()
            action = scheduler.next_action()
            if action is None:
                logger.debug("Phase '%s' has no scheduled actions; advancing.", phase_def.name)
                scheduler.advance("noop")
                continue

            phase_config = phase_def.to_mapping()

            if action == "collect":
                collect_result = self.workflow.collect_step(self.global_step, phase=phase_config)
                if collect_result is None or getattr(collect_result, "steps", 0) == 0:
                    logger.warning("Collect step returned no data for phase '%s'; stopping early.", phase_def.name)
                    break

                self._route_collect_result(collect_result)
                if collect_result.metrics:
                    self.experiment_logger.log_metrics(
                        collect_result.metrics,
                        self.global_step,
                        prefix="collect",
                    )
                self.global_step += int(collect_result.steps)
                self._update_context(global_step=self.global_step)
                scheduler.advance(action, steps=int(collect_result.steps))
                last_batch = None

            elif action == "update_world_model":
                data_source = self._resolve_data_source(phase_config)
                if data_source is None or not data_source.ready():
                    logger.debug(
                        "Skipping world-model update in phase '%s': data source unavailable or not ready.",
                        phase_def.name,
                    )
                    scheduler.advance(action)
                    continue

                batch = data_source.sample()
                last_batch = batch
                last_update_metrics = self.workflow.update_world_model(batch, phase=phase_config) or {}
                if last_update_metrics:
                    self.experiment_logger.log_metrics(last_update_metrics, self.global_step, prefix="train")
                scheduler.advance(action, updates=1)

            elif action == "update_controller":
                data_source = self._resolve_data_source(phase_config)
                if last_batch is None:
                    if data_source is None or not data_source.ready():
                        logger.debug(
                            "Skipping controller update in phase '%s': no batch available.",
                            phase_def.name,
                        )
                        scheduler.advance(action)
                        continue
                    last_batch = data_source.sample()
                controller_metrics = self.workflow.update_controller(last_batch, phase=phase_config) or {}
                if controller_metrics:
                    self.experiment_logger.log_metrics(controller_metrics, self.global_step, prefix="controller")
                scheduler.advance(action, updates=1)

            elif action == "evaluate":
                eval_metrics = self._run_evaluation_phase(phase_config)
                if eval_metrics:
                    self.experiment_logger.log_metrics(eval_metrics, self.global_step, prefix="eval")
                scheduler.advance(action)

            else:
                logger.warning("Encountered unknown scheduler action '%s'; advancing.", action)
                scheduler.advance(action)

            if log_freq > 0 and (self.global_step - last_log) >= log_freq:
                self.workflow.log_metrics(self.global_step, self.experiment_logger)
                last_log = self.global_step

            if ckpt_freq > 0 and (self.global_step - last_ckpt) >= ckpt_freq:
                self._save_checkpoint()
                last_ckpt = self.global_step

            if eval_freq > 0 and (self.global_step - last_eval) >= eval_freq:
                eval_metrics = self._run_evaluation_phase({"type": "eval_only", "name": "scheduled_eval"})
                if eval_metrics:
                    self.experiment_logger.log_metrics(eval_metrics, self.global_step, prefix="eval")
                last_eval = self.global_step

        duration = max(time.time() - start_time, 1e-6)
        final_metrics = dict(last_update_metrics)
        final_metrics.setdefault("wall_time", duration)
        self.workflow.log_metrics(self.global_step, self.experiment_logger)
        self._save_checkpoint(final=True)
        self.experiment_logger.finish()
        return final_metrics

    def train(self) -> Dict[str, float]:
        """Compatibility shim so orchestrator can be driven by legacy trainer interface."""
        return self.run()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_context(self, **updates: Any) -> None:
        context = self.ensure_context()
        new_context = context.with_updates(**updates)
        self._context = new_context
        self.workflow.on_context_update(new_context)
        if self.controller_manager is None and new_context.controller_manager is not None:
            self.controller_manager = new_context.controller_manager

    def _controller_state_dict(self) -> Dict[str, Any]:
        manager = self.controller_manager
        if manager is None:
            context = self._context or self.ensure_context()
            manager = context.controller_manager
        if manager is None:
            return {}
        return manager.state_dict()

    def _route_collect_result(self, result: CollectResult) -> None:
        extras = getattr(result, "extras", {}) or {}
        for source_name, payload in extras.items():
            data_source = self.data_sources.get(source_name)
            if data_source is None:
                logger.debug("No data source named '%s'; skipping payload.", source_name)
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if isinstance(item, dict):
                    data_source.add(**item)
                else:
                    data_source.add(trajectory=item)

    def _resolve_data_source(self, phase_config: Mapping[str, Any]) -> Optional[DataSource]:
        source_name = phase_config.get("data_source", "replay")
        return self.data_sources.get(source_name)

    def _run_evaluation_phase(self, phase_config: Mapping[str, Any]) -> Dict[str, float]:
        context = self.ensure_context()
        eval_environment = context.eval_environment
        if eval_environment is None:
            logger.warning("No evaluation environment available; skipping evaluation phase.")
            return {}

        eval_config = self._prepare_evaluation_config(phase_config)
        episodes_target = max(int(eval_config.get("num_episodes", 1) or 1), 1)
        max_episode_steps = eval_config.get("max_episode_steps")
        eval_phase_payload = {
            "type": eval_config.get("type", "eval_only"),
            "name": eval_config.get("name", "evaluation"),
            "deterministic": bool(eval_config.get("deterministic", True)),
            "controller_role": eval_config.get("controller_role"),
            "num_episodes": episodes_target,
            "max_episode_steps": max_episode_steps,
        }

        workflow = self.workflow
        original_state = {
            "environment": getattr(workflow, "environment", None),
            "is_vectorized": getattr(workflow, "is_vectorized", False),
            "num_envs": getattr(workflow, "num_envs", 1),
            "current_obs": getattr(workflow, "current_obs", None),
            "current_dones": getattr(workflow, "current_dones", None),
            "episode_returns": getattr(workflow, "episode_returns", None),
            "episode_lengths": getattr(workflow, "episode_lengths", None),
            "total_episodes": getattr(workflow, "total_episodes", 0),
            "current_episode_return": getattr(workflow, "current_episode_return", 0.0),
            "current_episode_length": getattr(workflow, "current_episode_length", 0),
            "vector_episode_returns": getattr(workflow, "vector_episode_returns", None),
            "vector_episode_lengths": getattr(workflow, "vector_episode_lengths", None),
        }

        maxlen_returns = getattr(original_state["episode_returns"], "maxlen", 100) or 100
        maxlen_lengths = getattr(original_state["episode_lengths"], "maxlen", 100) or 100

        workflow.environment = eval_environment
        workflow.is_vectorized = bool(getattr(eval_environment, "is_vectorized", False))
        workflow.num_envs = int(getattr(eval_environment, "num_envs", 1))
        workflow.current_obs = None
        workflow.current_dones = None
        workflow.episode_returns = deque(maxlen=maxlen_returns)
        workflow.episode_lengths = deque(maxlen=maxlen_lengths)
        workflow.total_episodes = 0
        workflow.current_episode_return = 0.0
        workflow.current_episode_length = 0
        if workflow.is_vectorized:
            workflow.vector_episode_returns = np.zeros(workflow.num_envs, dtype=np.float32)
            workflow.vector_episode_lengths = np.zeros(workflow.num_envs, dtype=np.int32)
        else:
            workflow.vector_episode_returns = None
            workflow.vector_episode_lengths = None

        start_time = time.time()

        collected_returns: list[float] = []
        collected_lengths: list[float] = []
        total_steps = 0
        total_iterations = 0

        safety_limit = max(episodes_target * 10, 10)

        try:
            while len(collected_returns) < episodes_target and total_iterations < safety_limit:
                result = workflow.collect_step(self.global_step, phase=eval_phase_payload)
                total_iterations += 1
                if result is None:
                    break

                steps = int(getattr(result, "steps", 0))
                total_steps += max(steps, 0)

                extras = getattr(result, "extras", {}) or {}
                eval_stats = extras.get("evaluation") or {}
                collected_returns.extend(eval_stats.get("returns", []))
                collected_lengths.extend(eval_stats.get("lengths", []))

                if max_episode_steps is not None and total_steps >= max_episode_steps * episodes_target:
                    break

                if steps <= 0 and not eval_stats:
                    # Avoid infinite loops if no progress is made.
                    break
        finally:
            workflow.environment = original_state["environment"]
            workflow.is_vectorized = original_state["is_vectorized"]
            workflow.num_envs = original_state["num_envs"]
            workflow.current_obs = original_state["current_obs"]
            workflow.current_dones = original_state["current_dones"]
            workflow.episode_returns = original_state["episode_returns"]
            workflow.episode_lengths = original_state["episode_lengths"]
            workflow.total_episodes = original_state["total_episodes"]
            workflow.current_episode_return = original_state["current_episode_return"]
            workflow.current_episode_length = original_state["current_episode_length"]
            workflow.vector_episode_returns = original_state["vector_episode_returns"]
            workflow.vector_episode_lengths = original_state["vector_episode_lengths"]

        duration = time.time() - start_time

        if len(collected_returns) > episodes_target:
            collected_returns = collected_returns[:episodes_target]
        if len(collected_lengths) > episodes_target:
            collected_lengths = collected_lengths[:episodes_target]

        metrics: Dict[str, Any] = {
            "phase": eval_config.get("name", "evaluation"),
            "step": float(self.global_step),
            "episodes": float(len(collected_returns)),
            "episodes_target": float(episodes_target),
            "steps": float(total_steps),
            "duration": float(duration),
        }

        if collected_returns:
            returns_array = np.asarray(collected_returns, dtype=np.float32)
            metrics.update(
                {
                    "return_mean": float(returns_array.mean()),
                    "return_std": float(returns_array.std()),
                    "return_max": float(returns_array.max()),
                    "return_min": float(returns_array.min()),
                }
            )
        else:
            metrics.update(
                {
                    "return_mean": float("nan"),
                    "return_std": float("nan"),
                    "return_max": float("nan"),
                    "return_min": float("nan"),
                }
            )

        if collected_lengths:
            lengths_array = np.asarray(collected_lengths, dtype=np.float32)
            metrics["episode_length_mean"] = float(lengths_array.mean())
        else:
            metrics["episode_length_mean"] = float("nan")

        normalized = self._normalize_metrics(metrics)
        normalized.setdefault("phase", str(eval_config.get("name", "evaluation")))
        normalized.setdefault("step", float(self.global_step))
        normalized.setdefault("duration", float(duration))
        return normalized

    def _prepare_evaluation_config(self, phase_config: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(phase_config or {})
        nested = payload.pop("evaluation", None)
        if isinstance(nested, Mapping):
            payload.update(nested)

        name = str(payload.get("name", payload.get("type", "evaluation")))
        training_cfg = getattr(self.config, "training", None)
        default_episodes = 1
        default_max_steps = None
        default_controller = None

        if training_cfg is not None:
            default_episodes = int(getattr(training_cfg, "num_eval_episodes", default_episodes) or default_episodes)
            default_max_steps = getattr(training_cfg, "max_eval_steps", None)
            default_controller = getattr(training_cfg, "eval_controller_role", None)

        controller_role = payload.get("controller_role", default_controller)
        if controller_role is None and "planner" in self.controllers:
            controller_role = "planner"

        return {
            "name": name,
            "type": payload.get("type", "eval_only"),
            "num_episodes": int(payload.get("num_episodes", default_episodes)),
            "deterministic": bool(payload.get("deterministic", True)),
            "controller_role": controller_role,
            "max_episode_steps": payload.get("max_episode_steps", default_max_steps),
        }

    def _normalize_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, Number):
                normalized[key] = float(value)
            elif isinstance(value, np.generic):
                normalized[key] = float(value.item())
            elif isinstance(value, np.ndarray) and value.size == 1:
                normalized[key] = float(value.reshape(()).item())
        return normalized

    def _save_checkpoint(self, *, final: bool = False) -> None:
        state = {
            "workflow_name": self.workflow.__class__.__name__,
            "buffer": self.ensure_context().buffer,
            "metrics": {},
            "world_model_updates": getattr(self.workflow, "world_model_updates", 0),
            "workflow": self.workflow.state_dict(),
            "data_sources": {name: src.state_dict() for name, src in self.data_sources.items()},
            "controllers": self._controller_state_dict(),
        }
        name = "final" if final else None
        self.checkpoint_manager.save_checkpoint(state, self.global_step, name=name, is_best=final)

    def save_experiment_config(self) -> None:
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        save_config(self.config, config_dir / "config.yaml")

    def cleanup(self) -> None:
        context = self._context
        if context is None:
            return
        try:
            context.train_environment.close()
        except Exception:
            pass
        if context.eval_environment is not context.train_environment:
            try:
                context.eval_environment.close()
            except Exception:
                pass
