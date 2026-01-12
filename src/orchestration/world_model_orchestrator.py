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

        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            cwd = Path.cwd()
            # When Hydra chdir is enabled, use the run directory as the experiment root.
            if (cwd / ".hydra").exists():
                self.experiment_dir = cwd
            else:
                timestamp = getattr(config.experiment, "timestamp", None)
                if timestamp is None:
                    from datetime import datetime

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.experiment_dir = Path("experiments") / f"{config.experiment.name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
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
        try:
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

                phase_config = phase_def.to_mapping(progress=scheduler.get_state())

                if action == "collect":
                    collect_result = self.workflow.collect_step(self.global_step, phase=phase_config)
                    if collect_result is None or getattr(collect_result, "steps", 0) == 0:
                        logger.warning("Collect step returned no data for phase '%s'; stopping early.", phase_def.name)
                        break

                    self._route_collect_result(collect_result, phase_config)
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
                    self.global_step += 1
                    self._update_context(global_step=self.global_step)

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
                    self.global_step += 1
                    self._update_context(global_step=self.global_step)

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
        finally:
            self.cleanup()
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

    def _route_collect_result(self, result: CollectResult, phase_config: Mapping[str, Any]) -> None:
        """Route collected trajectory to the buffer specified in phase config.

        Args:
            result: CollectResult from workflow containing trajectory data
            phase_config: Phase configuration containing buffer name
        """
        buffer = self._resolve_buffer(phase_config)
        if buffer is None:
            logger.debug("No buffer configured for phase '%s'; skipping collected data.", phase_config.get("name"))
            return

        if result.trajectory is not None:
            buffer.add(trajectory=result.trajectory)

    def _resolve_buffer(self, phase_config: Mapping[str, Any]) -> Optional[Any]:
        buffer_name = phase_config.get("buffer", "replay")
        return self.buffers.get(buffer_name)

    def _run_evaluation_phase(self, phase_config: Mapping[str, Any]) -> Dict[str, float]:
        raise NotImplementedError("Evaluation phase not implemented")
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
        raise NotImplementedError("Evaluation phase not implemented")
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
        """Save checkpoint with loop-based component/optimizer discovery."""
        context = self.ensure_context()

        # 1. Collect component states (loop-based discovery)
        components_state = {}
        for name, component in context.components.components.items():
            if component is not None and isinstance(component, torch.nn.Module):
                components_state[name] = component.state_dict()

        # 2. Collect optimizer states (loop-based discovery)
        optimizers_state = {}
        for name, optimizer in (context.optimizers or {}).items():
            if optimizer is not None:
                optimizers_state[name] = optimizer.state_dict()

        # 3. Collect controller states
        controllers_state = {}
        if self.controller_manager is not None:
            controllers_state = self.controller_manager.state_dict()

        # 4. Collect phase scheduler state
        phase_state = self.scheduler.get_state()

        # 5. Collect workflow custom state
        workflow_custom = self.workflow.get_state()

        # 6. Assemble checkpoint (buffers rely on finalize() for persistence)
        state = {
            "version": 1,
            "global_step": self.global_step,
            "workflow_name": self.workflow.__class__.__name__,
            "phase_state": phase_state,
            "components": components_state,
            "controllers": controllers_state,
            "optimizers": optimizers_state,
            "workflow_custom": workflow_custom,
        }

        # 8. Delegate I/O to checkpoint manager
        name = "final" if final else f"step_{self.global_step}"
        self.checkpoint_manager.save(state, self.global_step, name=name, is_best=final)

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint and restore all state."""
        # 1. Load raw checkpoint
        ckpt = self.checkpoint_manager.load(path)
        if ckpt is None:
            logger.warning(f"No checkpoint found at {path}")
            return

        context = self.ensure_context()

        # 2. Restore global step
        self.global_step = ckpt.get("global_step", 0)
        logger.info(f"Restoring checkpoint from step {self.global_step}")

        # 3. Restore phase scheduler state
        phase_state = ckpt.get("phase_state", {})
        self.scheduler.set_state(phase_state)

        # 4. Restore component weights (loop-based)
        components_state = ckpt.get("components", {})
        for name, state_dict in components_state.items():
            component = context.components.components.get(name)
            if component is not None and isinstance(component, torch.nn.Module):
                component.load_state_dict(state_dict)
                logger.debug(f"Restored component: {name}")

        # 5. Restore optimizer states with LR override from config
        optimizers_state = ckpt.get("optimizers", {})
        for name, opt_state in optimizers_state.items():
            optimizer = (context.optimizers or {}).get(name)
            if optimizer is not None:
                optimizer.load_state_dict(opt_state)
                # Override LR from config
                lr_key = f"{name}_lr"
                config_lr = getattr(self.config.algorithm, lr_key, None)
                if config_lr is not None:
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(config_lr)
                    logger.debug(f"Restored optimizer {name} with LR override: {config_lr}")
                else:
                    logger.debug(f"Restored optimizer {name} (no LR override)")

        # 6. Restore controller states
        controllers_state = ckpt.get("controllers", {})
        if self.controller_manager is not None and controllers_state:
            self.controller_manager.load_state_dict(controllers_state)
            logger.debug("Restored controller states")

        # 7. Restore workflow custom state
        workflow_custom = ckpt.get("workflow_custom", {})
        self.workflow.set_state(workflow_custom)
        logger.debug("Restored workflow custom state")

        # 8. Update context global_step (buffers rely on finalize() for persistence)
        self._update_context(global_step=self.global_step)

        # 9. Restore RNG states
        self.checkpoint_manager.restore_rng_states(ckpt)

        logger.info(f"Checkpoint restored successfully from step {self.global_step}")

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

        # Finalize buffers (writes metadata for disk buffers)
        if context.buffers:
            for buffer in context.buffers.values():
                if hasattr(buffer, "finalize") and callable(buffer.finalize):
                    try:
                        buffer.finalize()
                    except Exception:
                        pass

        try:
            context.train_environment.close()
        except Exception:
            pass
        if context.eval_environment is not context.train_environment:
            try:
                context.eval_environment.close()
            except Exception:
                pass
