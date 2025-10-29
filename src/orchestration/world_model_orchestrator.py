"""Orchestrator coordinating world-model workflows."""

from __future__ import annotations

import copy
import logging
import random
import time
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config, resolve_device, save_config
from ..utils.logger import create_logger
from ..workflows.world_models.base import CollectResult, WorldModelWorkflow
from ..workflows.world_models.context import WorkflowContext, WorldModelComponents
from ..workflows.world_models.controllers import ControllerManager
from .phase_scheduler import PhaseScheduler, PhaseDefinition

logger = logging.getLogger(__name__)


def _to_dict(payload: Any) -> Dict[str, Any]:
    """Best-effort conversion of config-like objects to plain dictionaries."""
    if hasattr(payload, "to_dict"):
        return payload.to_dict()  # type: ignore[return-value]
    if DictConfig is not None and isinstance(payload, DictConfig):
        return OmegaConf.to_container(payload, resolve=True)  # type: ignore[return-value]
    if isinstance(payload, Mapping):
        return copy.deepcopy(dict(payload))
    if hasattr(payload, "__dict__"):
        return copy.deepcopy(payload.__dict__)
    raise TypeError(f"Cannot convert payload of type {type(payload)} to dict.")


class WorldModelOrchestrator:
    """Owns experiment lifecycle while delegating algorithm detail to workflows."""

    def __init__(
        self,
        config: Config | DictConfig | Mapping[str, Any],
        workflow: WorldModelWorkflow,
        *,
        experiment_dir: Optional[Path] = None,
        scheduler: Optional["PhaseScheduler"] = None,
        components: Optional[WorldModelComponents] = None,
        optimizers: Optional[Dict[str, Any]] = None,
        train_environment: Optional[Any] = None,
        eval_environment: Optional[Any] = None,
        buffers: Optional[Dict[str, Any]] = None,
        controllers: Optional[Dict[str, Any]] = None,
        controller_manager: Optional[ControllerManager] = None,
    ) -> None:
        self.config = config
        self.workflow = workflow
        self.device = resolve_device(config.experiment.device)
        self.scheduler = scheduler or self._build_scheduler()
        self.buffers: Dict[str, Any] = dict(buffers or {})
        self.controller_manager: Optional[ControllerManager] = controller_manager
        self._initial_resources: Optional[Dict[str, Any]] = {
            "components": components,
            "optimizers": optimizers,
            "train_environment": train_environment,
            "eval_environment": eval_environment,
            "buffers": buffers,
            "controllers": controllers,
            "controller_manager": controller_manager,
        }
        self.global_step = 0

        timestamp = getattr(config.experiment, "timestamp", None)
        if timestamp is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        default_dir = Path("experiments") / f"{config.experiment.name}_{timestamp}"
        self.experiment_dir = Path(experiment_dir) if experiment_dir else default_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
            auto_save_frequency=self.config.training.checkpoint_frequency,
            max_checkpoints=5,
        )
        self.experiment_logger = create_logger(
            self.experiment_dir,
            _to_dict(getattr(self.config, "logging", {})),
            _to_dict(self.config),
        )

        self._context: Optional[WorkflowContext] = None
        self._initialized = False

    def ensure_context(self) -> WorkflowContext:
        if self._context is None:
            if self._initial_resources is None:
                raise RuntimeError("Orchestrator context requested before resources were provided.")
            self._context = self._build_context(self._initial_resources)
            self._initial_resources = None
            self.global_step = self._context.global_step
        return self._context

    def _build_scheduler(self) -> PhaseScheduler:
        raw_phases = getattr(self.config.training, "phases", None)
        phases_config: Optional[list[Dict[str, Any]]] = None

        if raw_phases:
            phases_config = []
            for entry in raw_phases:
                try:
                    phases_config.append(_to_dict(entry))
                except TypeError:
                    phases_config.append(copy.deepcopy(getattr(entry, "__dict__", {})))

        return PhaseScheduler(phases_config)

    def _build_context(self, resources: Dict[str, Any]) -> WorkflowContext:
        logger.info("Building workflow context for orchestrator path")

        train_environment = resources.get("train_environment")
        if train_environment is None:
            raise RuntimeError("WorldModelOrchestrator requires a pre-instantiated training environment.")
        eval_environment = resources.get("eval_environment") or train_environment

        seed = getattr(getattr(self.config, "experiment", None), "seed", None)
        self._configure_seeds(seed)

        components_input = resources.get("components")
        if components_input is None:
            raise RuntimeError("WorldModelOrchestrator requires pre-instantiated world model components.")
        components = (
            components_input
            if isinstance(components_input, WorldModelComponents)
            else WorldModelComponents(**components_input)  # type: ignore[arg-type]
        )
        components.to(self.device)

        optimizers_input = resources.get("optimizers") or {}
        if not optimizers_input:
            raise RuntimeError("WorldModelOrchestrator requires optimizer mappings aligned with the provided components.")
        optimizers = dict(optimizers_input)

        buffers_input = resources.get("buffers") or {}
        buffers: Dict[str, Any] = {name: buf for name, buf in buffers_input.items() if buf is not None}
        if not buffers:
            raise RuntimeError("WorldModelOrchestrator requires at least one buffer instance.")

        context = WorkflowContext(
            config=self.config,
            device=self.device,
            train_environment=train_environment,
            eval_environment=eval_environment,
            components=components,
            checkpoint_manager=self.checkpoint_manager,
            experiment_logger=self.experiment_logger,
            buffers=buffers,
        )

        initial_obs = None
        initial_dones = None
        if hasattr(train_environment, "reset"):
            reset_kwargs: Dict[str, Any] = {}
            if seed is not None:
                reset_kwargs["seed"] = seed
            try:
                initial_obs = train_environment.reset(**reset_kwargs)
            except TypeError:
                initial_obs = train_environment.reset()
            if isinstance(initial_obs, tuple):
                initial_obs = initial_obs[0]
            num_envs = int(getattr(train_environment, "num_envs", 1) or 1)
            initial_dones = np.zeros(num_envs, dtype=bool)

        controller_manager = resources.get("controller_manager")
        controllers_input = resources.get("controllers")
        if controller_manager is None and controllers_input is not None:
            controller_manager = ControllerManager(dict(controllers_input))
        if controller_manager is None:
            raise RuntimeError("WorldModelOrchestrator requires instantiated controllers or a controller manager.")
        controllers = dict(controller_manager.as_dict())

        context = context.with_updates(
            controllers=controllers or None,
            controller_manager=controller_manager,
        )
        self.controller_manager = controller_manager
        buffers = self._initialize_buffers(context, buffers)
        context = context.with_updates(buffers=buffers)
        self.buffers = buffers

        context = context.with_updates(
            optimizers=optimizers,
            initial_observation=initial_obs,
            initial_dones=initial_dones,
        )

        return context

    def _configure_seeds(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_buffers(
        self,
        context: WorkflowContext,
        buffers: Dict[str, Any],
    ) -> Dict[str, Any]:
        initialised: Dict[str, Any] = {}
        for name, buffer in buffers.items():
            if buffer is None:
                continue
            initializer = getattr(buffer, "initialize", None)
            if callable(initializer):
                try:
                    initializer(context)
                except TypeError:
                    initializer()
            initialised[name] = buffer
        return initialised

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
                buffer = self._resolve_buffer(phase_config)
                if buffer is None or not getattr(buffer, "ready", lambda: True)():
                    logger.debug(
                        "Skipping world-model update in phase '%s': buffer unavailable or not ready.",
                        phase_def.name,
                    )
                    scheduler.advance(action)
                    continue

                batch = buffer.sample()
                last_batch = batch
                last_update_metrics = self.workflow.update_world_model(batch, phase=phase_config) or {}
                if last_update_metrics:
                    self.experiment_logger.log_metrics(last_update_metrics, self.global_step, prefix="train")
                scheduler.advance(action, updates=1)

            elif action == "update_controller":
                buffer = self._resolve_buffer(phase_config)
                if last_batch is None:
                    if buffer is None or not getattr(buffer, "ready", lambda: True)():
                        logger.debug(
                            "Skipping controller update in phase '%s': no batch available.",
                            phase_def.name,
                        )
                        scheduler.advance(action)
                        continue
                    last_batch = buffer.sample()
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
                workflow_metrics = self.workflow.state_dict(mode="metrics")
                if workflow_metrics:
                    self.experiment_logger.log_metrics(workflow_metrics, self.global_step, prefix="workflow")
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
        workflow_metrics = self.workflow.state_dict(mode="metrics")
        if workflow_metrics:
            self.experiment_logger.log_metrics(workflow_metrics, self.global_step, prefix="workflow")
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

    def _controller_state_dict(self, *, mode: str = "checkpoint") -> Dict[str, Any]:
        manager = self.controller_manager
        if manager is None:
            context = self._context or self.ensure_context()
            manager = context.controller_manager
        if manager is None:
            return {}
        return manager.state_dict(mode=mode)

    def _route_collect_result(self, result: CollectResult) -> None:
        extras = getattr(result, "extras", {}) or {}
        for source_name, payload in extras.items():
            buffer = self.buffers.get(source_name)
            if buffer is None:
                logger.debug("No buffer named '%s'; skipping payload.", source_name)
                continue
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if isinstance(item, dict):
                    buffer.add(**item)
                else:
                    buffer.add(trajectory=item)

    def _resolve_buffer(self, phase_config: Mapping[str, Any]) -> Optional[Any]:
        buffer_name = phase_config.get("buffer", "replay")
        return self.buffers.get(buffer_name)

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
        episode_snapshot = workflow._snapshot_episode_tracking()
        original_state = {
            "environment": getattr(workflow, "environment", None),
            "is_vectorized": getattr(workflow, "is_vectorized", False),
            "num_envs": getattr(workflow, "num_envs", 1),
            "current_obs": getattr(workflow, "current_obs", None),
            "current_dones": getattr(workflow, "current_dones", None),
            "episode_tracking": episode_snapshot,
        }

        workflow.environment = eval_environment
        workflow.is_vectorized = bool(getattr(eval_environment, "is_vectorized", False))
        workflow.num_envs = int(getattr(eval_environment, "num_envs", 1))
        workflow.current_obs = None
        workflow.current_dones = None
        workflow._reset_episode_tracking(
            workflow.num_envs,
            clear_history=True,
            history=episode_snapshot.get("history_len"),
        )

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
            workflow._restore_episode_tracking(original_state["episode_tracking"])

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
        manager = self.controller_manager
        if controller_role is None and manager is not None and "planner" in set(manager.roles()):
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
            "metrics": {},
            "world_model_updates": getattr(self.workflow, "world_model_updates", 0),
            "workflow": self.workflow.state_dict(mode="checkpoint"),
            "buffers": {name: self._buffer_checkpoint(buf) for name, buf in self.buffers.items()},
            "controllers": self._controller_state_dict(),
        }
        name = "final" if final else None
        self.checkpoint_manager.save_checkpoint(state, self.global_step, name=name, is_best=final)

    def save_experiment_config(self) -> None:
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        target_path = config_dir / "config.yaml"
        if DictConfig is not None and isinstance(self.config, DictConfig):
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write(OmegaConf.to_yaml(self.config))
        else:
            save_config(self.config, target_path)

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

    def _buffer_checkpoint(self, buffer: Any) -> Dict[str, Any]:
        saver = getattr(buffer, "save_checkpoint", None)
        if callable(saver):
            try:
                return saver()
            except TypeError:
                return saver
        state_dict = getattr(buffer, "state_dict", None)
        if callable(state_dict):
            try:
                return state_dict(mode="checkpoint")
            except TypeError:
                return state_dict()
        return {}
