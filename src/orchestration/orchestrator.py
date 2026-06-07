"""Orchestrator coordinating workflows."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import socket
import sys
import time
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch

from ..utils.checkpoint import CheckpointManager
from ..utils.config import Config, resolve_device, save_config
from ..utils.logger import create_logger
from ..workflows.utils.base import CollectResult, WorldModelWorkflow
from ..workflows.utils.context import WorkflowContext, WorldModelComponents
from ..workflows.utils.controllers import ControllerManager
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


class Orchestrator:
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
        self.best_eval_return = float("-inf")

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
        self._run_status_path = self.experiment_dir / "run_status.json"
        self._run_started_at = self._utc_now()
        self._run_status_phase: Optional[str] = None
        self._run_status_action: Optional[str] = None
        self._run_status_hook_state: Optional[str] = None
        self._run_status_metrics: Dict[str, float] = {}
        self._checkpoint_name_counts: Dict[str, int] = {}

        training_cfg = getattr(self.config, "training", None)
        max_checkpoints = 5
        if training_cfg is not None:
            configured_max_checkpoints = getattr(training_cfg, "max_checkpoints", None)
            if configured_max_checkpoints is not None:
                max_checkpoints = int(configured_max_checkpoints)

        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
            max_checkpoints=max_checkpoints,
        )
        self.experiment_logger = create_logger(
            self.experiment_dir,
            _to_dict(getattr(self.config, "logging", {})),
            _to_dict(self.config),
        )

        self._context: Optional[WorkflowContext] = None
        self._initialized = False
        self._write_run_status(status="initializing")

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
            raise RuntimeError("Orchestrator requires a pre-instantiated training environment.")
        eval_environment = resources.get("eval_environment") or train_environment

        seed = getattr(getattr(self.config, "experiment", None), "seed", None)
        self._configure_seeds(seed)

        components_input = resources.get("components")
        if components_input is None:
            raise RuntimeError("Orchestrator requires pre-instantiated world model components.")
        components = (
            components_input
            if isinstance(components_input, WorldModelComponents)
            else WorldModelComponents(**components_input)  # type: ignore[arg-type]
        )
        components.to(self.device)

        optimizers_input = resources.get("optimizers") or {}
        if not optimizers_input:
            raise RuntimeError("Orchestrator requires optimizer mappings aligned with the provided components.")
        optimizers = dict(optimizers_input)

        buffers_input = resources.get("buffers") or {}
        buffers: Dict[str, Any] = {name: buf for name, buf in buffers_input.items() if buf is not None}
        if not buffers:
            raise RuntimeError("Orchestrator requires at least one buffer instance.")

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
            raise RuntimeError("Orchestrator requires instantiated controllers or a controller manager.")
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
            self._write_run_status(status="running", hook_state="starting")
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
                self._write_run_status(
                    status="running",
                    phase=phase_def.name,
                    action=action,
                    hook_state="running",
                )

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
                    self._write_run_status(
                        status="running",
                        phase=phase_def.name,
                        action=action,
                        hook_state="completed",
                        metrics=self._prefix_metrics(collect_result.metrics, "collect"),
                    )

                elif action == "update_world_model":
                    buffer = self._resolve_buffer(phase_config)
                    if buffer is None or not getattr(buffer, "ready", lambda: True)():
                        logger.debug(
                            "Skipping world-model update in phase '%s': buffer unavailable or not ready.",
                            phase_def.name,
                        )
                        scheduler.advance(action)
                        self._write_run_status(
                            status="running",
                            phase=phase_def.name,
                            action=action,
                            hook_state="skipped",
                        )
                        continue

                    batch = buffer.sample()
                    last_batch = batch
                    last_update_metrics = self.workflow.update_world_model(batch, phase=phase_config) or {}
                    if last_update_metrics:
                        self.experiment_logger.log_metrics(last_update_metrics, self.global_step, prefix="train")
                    scheduler.advance(action, updates=1)
                    self.global_step += 1
                    self._update_context(global_step=self.global_step)
                    self._write_run_status(
                        status="running",
                        phase=phase_def.name,
                        action=action,
                        hook_state="completed",
                        metrics=self._prefix_metrics(last_update_metrics, "train"),
                    )

                elif action == "update_controller":
                    buffer = self._resolve_buffer(phase_config)
                    if buffer is None or not getattr(buffer, "ready", lambda: True)():
                        logger.debug(
                            "Skipping controller update in phase '%s': no batch available.",
                            phase_def.name,
                        )
                        scheduler.advance(action)
                        self._write_run_status(
                            status="running",
                            phase=phase_def.name,
                            action=action,
                            hook_state="skipped",
                        )
                        continue
                    batch = buffer.sample()
                    controller_metrics = self.workflow.update_controller(batch, phase=phase_config) or {}
                    if controller_metrics:
                        self.experiment_logger.log_metrics(controller_metrics, self.global_step, prefix="controller")
                    scheduler.advance(action, updates=1)
                    self.global_step += 1
                    self._update_context(global_step=self.global_step)
                    self._write_run_status(
                        status="running",
                        phase=phase_def.name,
                        action=action,
                        hook_state="completed",
                        metrics=self._prefix_metrics(controller_metrics, "controller"),
                    )

                elif action == "evaluate":
                    eval_metrics = self._run_evaluation(phase_config)
                    if eval_metrics:
                        self.experiment_logger.log_metrics(eval_metrics, self.global_step, prefix="eval")
                        self._maybe_save_best_checkpoint(eval_metrics)
                    scheduler.advance(action)
                    self._write_run_status(
                        status="running",
                        phase=phase_def.name,
                        action=action,
                        hook_state="completed",
                        metrics=self._prefix_metrics(eval_metrics, "eval"),
                    )

                else:
                    logger.warning("Encountered unknown scheduler action '%s'; advancing.", action)
                    scheduler.advance(action)
                    self._write_run_status(
                        status="running",
                        phase=phase_def.name,
                        action=action,
                        hook_state="skipped",
                    )

                if log_freq > 0 and (self.global_step - last_log) >= log_freq:
                    workflow_metrics = self.workflow.get_state()
                    if workflow_metrics:
                        self.experiment_logger.log_metrics(workflow_metrics, self.global_step, prefix="workflow")
                    last_log = self.global_step

                if ckpt_freq > 0 and (self.global_step - last_ckpt) >= ckpt_freq:
                    self._save_checkpoint()
                    last_ckpt = self.global_step

                if eval_freq > 0 and (self.global_step - last_eval) >= eval_freq:
                    self._write_run_status(
                        status="running",
                        phase="scheduled_eval",
                        action="evaluate",
                        hook_state="running",
                    )
                    eval_metrics = self._run_evaluation({"type": "eval_only", "name": "scheduled_eval"})
                    if eval_metrics:
                        self.experiment_logger.log_metrics(eval_metrics, self.global_step, prefix="eval")
                        self._maybe_save_best_checkpoint(eval_metrics)
                    self._write_run_status(
                        status="running",
                        phase="scheduled_eval",
                        action="evaluate",
                        hook_state="completed",
                        metrics=self._prefix_metrics(eval_metrics, "eval"),
                    )
                    last_eval = self.global_step

            duration = max(time.time() - start_time, 1e-6)
            final_metrics = dict(last_update_metrics)
            final_metrics.setdefault("wall_time", duration)
            workflow_metrics = self.workflow.get_state()
            if workflow_metrics:
                self.experiment_logger.log_metrics(workflow_metrics, self.global_step, prefix="workflow")
            final_checkpoint = self._save_checkpoint(final=True)
            self._write_run_summary(
                status="completed",
                final_metrics=final_metrics,
                workflow_metrics=workflow_metrics,
                final_checkpoint=final_checkpoint,
            )
            self.experiment_logger.finish()
            self._write_run_status(status="completed")
        except Exception as exc:
            self._write_run_summary(status="failed", error=str(exc))
            self._write_run_status(status="failed", error=str(exc))
            raise
        finally:
            self.cleanup()
        return final_metrics

    def train(self) -> Dict[str, float]:
        """Run training through the public trainer-facing entrypoint."""
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

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _prefix_metrics(self, metrics: Optional[Mapping[str, Any]], prefix: str) -> Dict[str, float]:
        prefixed: Dict[str, float] = {}
        for key, value in (metrics or {}).items():
            if not isinstance(value, Number) and not isinstance(value, np.generic):
                continue
            metric_key = str(key)
            if prefix and not metric_key.startswith(prefix + "/"):
                metric_key = f"{prefix}/{metric_key}"
            prefixed[metric_key] = float(value)
        return prefixed

    def _write_run_status(
        self,
        *,
        status: str,
        phase: Optional[str] = None,
        action: Optional[str] = None,
        hook_state: Optional[str] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write a small machine-readable status snapshot for polling agents."""
        if phase is not None:
            self._run_status_phase = phase
        if action is not None:
            self._run_status_action = action
        if hook_state is not None:
            self._run_status_hook_state = hook_state
        for key, value in self._prefix_metrics(metrics, "").items():
            self._run_status_metrics[key] = value

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.experiment_dir.name,
            "status": status,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "command": " ".join(sys.argv),
            "experiment_dir": str(self.experiment_dir),
            "workflow_name": self.workflow.__class__.__name__,
            "global_step": int(self.global_step),
            "phase": self._run_status_phase,
            "action": self._run_status_action,
            "hook_state": self._run_status_hook_state,
            "started_at": self._run_started_at,
            "updated_at": self._utc_now(),
            "last_metrics": dict(sorted(self._run_status_metrics.items())),
        }
        if error is not None:
            payload["error"] = error

        tmp_path = self._run_status_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self._run_status_path)
        except Exception as exc:
            logger.debug("Failed to write run status: %s", exc)

    def _checkpoint_target(self, name: str) -> Optional[str]:
        path = self.checkpoint_manager.checkpoint_dir / name
        if not path.exists() and not path.is_symlink():
            return None
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def _write_run_summary(
        self,
        *,
        status: str,
        final_metrics: Optional[Mapping[str, Any]] = None,
        workflow_metrics: Optional[Mapping[str, Any]] = None,
        final_checkpoint: Optional[Path] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write final run facts for post-run inspection and manifest updates."""
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.experiment_dir.name,
            "status": status,
            "workflow_name": self.workflow.__class__.__name__,
            "experiment_dir": str(self.experiment_dir),
            "global_step": int(self.global_step),
            "best_eval_return": (
                float(self.best_eval_return)
                if np.isfinite(self.best_eval_return)
                else None
            ),
            "started_at": self._run_started_at,
            "completed_at": self._utc_now(),
            "final_metrics": self._normalize_metrics(final_metrics or {}),
            "workflow_metrics": self._normalize_metrics(workflow_metrics or {}),
            "checkpoints": {
                "final": str(final_checkpoint) if final_checkpoint is not None else self._checkpoint_target("final.pt"),
                "latest": self._checkpoint_target("latest.pt"),
                "best": self._checkpoint_target("best.pt"),
            },
        }
        if error is not None:
            payload["error"] = error

        summary_path = self.experiment_dir / "run_summary.json"
        tmp_path = summary_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(summary_path)
        except Exception as exc:
            logger.debug("Failed to write run summary: %s", exc)

    def _run_evaluation(self, phase_config: Mapping[str, Any]) -> Dict[str, float]:
        """Run evaluation through the workflow-owned evaluation interface."""
        workflow = self.workflow

        if not hasattr(workflow, "evaluate") or not callable(workflow.evaluate):
            raise NotImplementedError(
                f"{workflow.__class__.__name__} must implement evaluate() when evaluation is configured."
            )

        training_cfg = getattr(self.config, "training", None)
        total_episodes = 10
        max_steps = 1000

        if training_cfg is not None:
            total_episodes = int(getattr(training_cfg, "num_eval_episodes", total_episodes) or total_episodes)
            max_steps = int(getattr(training_cfg, "max_eval_steps", max_steps) or max_steps)

        eval_batches, eval_num_envs, eval_total_episodes = self._resolve_eval_workload(total_episodes)
        logger.info(
            "Running evaluation: %s total episodes as %s batch(es) across %s env(s)",
            eval_total_episodes,
            eval_batches,
            eval_num_envs,
        )
        metrics = workflow.evaluate(
            num_eval_batches=eval_batches,
            max_steps_per_episode=max_steps,
            deterministic=True,
        )
        metrics = dict(metrics or {})
        metrics.setdefault("episodes", float(eval_total_episodes))
        metrics.setdefault("eval_episode_batches", float(eval_batches))
        metrics.setdefault("eval_num_envs", float(eval_num_envs))
        metrics.setdefault("eval_total_episodes", float(eval_total_episodes))
        mean_return = metrics.get("return_mean")
        if isinstance(mean_return, Number):
            logger.info("Evaluation complete: mean_return=%.2f", float(mean_return))
        else:
            logger.info("Evaluation complete")
        return metrics

    def _resolve_eval_workload(self, total_episodes: int) -> tuple[int, int, int]:
        """Convert configured total eval episodes into vector-env batches."""
        total_episodes = int(total_episodes)
        if total_episodes <= 0:
            raise ValueError("training.num_eval_episodes must be positive.")

        context = self.ensure_context()
        eval_environment = context.eval_environment or context.train_environment
        eval_num_envs = int(getattr(eval_environment, "num_envs", 1) or 1)
        if eval_num_envs <= 0:
            raise ValueError("Evaluation environment num_envs must be positive.")
        if total_episodes % eval_num_envs != 0:
            raise ValueError(
                "training.num_eval_episodes must be divisible by the eval environment num_envs "
                f"({total_episodes} requested total episodes, {eval_num_envs} envs)."
            )
        return total_episodes // eval_num_envs, eval_num_envs, total_episodes

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

    def _maybe_save_best_checkpoint(self, metrics: Mapping[str, Any]) -> None:
        value = metrics.get("return_mean")
        if value is None:
            return
        try:
            return_mean = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(return_mean):
            return
        if return_mean <= self.best_eval_return:
            return

        self.best_eval_return = return_mean
        logger.info(
            "New best eval return %.2f at step %s; saving best checkpoint.",
            return_mean,
            self.global_step,
        )
        self._save_checkpoint(
            name=self._next_checkpoint_name(f"best_step_{self.global_step}"),
            is_best=True,
        )

    def _next_checkpoint_name(self, base_name: str) -> str:
        count = self._checkpoint_name_counts.get(base_name, 0)
        while True:
            candidate = base_name if count == 0 else f"{base_name}_{count}"
            path = self.checkpoint_manager.checkpoint_dir / f"{candidate}.pt"
            if not path.exists() and not path.is_symlink():
                self._checkpoint_name_counts[base_name] = count + 1
                return candidate
            count += 1

    def _save_checkpoint(
        self,
        *,
        final: bool = False,
        name: Optional[str] = None,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint with loop-based component/optimizer discovery."""
        context = self.ensure_context()

        # 1. Collect component states (loop-based discovery)
        components_state = {}
        for component_name, component in context.components.components.items():
            if component is not None and isinstance(component, torch.nn.Module):
                components_state[component_name] = component.state_dict()

        # 2. Collect optimizer states (loop-based discovery)
        optimizers_state = {}
        for optimizer_name, optimizer in (context.optimizers or {}).items():
            if optimizer is not None:
                optimizers_state[optimizer_name] = optimizer.state_dict()

        # 3. Collect controller states
        controllers_state = {}
        if self.controller_manager is not None:
            controllers_state = self.controller_manager.state_dict()

        # 4. Collect phase scheduler state
        phase_state = self.scheduler.get_state()

        # 5. Collect workflow custom state
        workflow_custom = self.workflow.get_state()

        # 6. Collect buffer state. Exact resume is only honest when replay state
        # can be restored from the checkpoint, not just from optimizer/model state.
        buffers_state = self._collect_buffer_states(context.buffers)

        # 7. Assemble checkpoint
        state = {
            "version": 1,
            "global_step": self.global_step,
            "workflow_name": self.workflow.__class__.__name__,
            "phase_state": phase_state,
            "components": components_state,
            "controllers": controllers_state,
            "optimizers": optimizers_state,
            "workflow_custom": workflow_custom,
            "buffers": buffers_state,
            "best_eval_return": self.best_eval_return,
        }

        # 8. Delegate I/O to checkpoint manager
        checkpoint_name = name or ("final" if final else f"step_{self.global_step}")
        return self.checkpoint_manager.save(state, self.global_step, name=checkpoint_name, is_best=is_best)

    def _collect_buffer_states(self, buffers: Mapping[str, Any]) -> Dict[str, Any]:
        states: Dict[str, Any] = {}
        for name, buffer in buffers.items():
            get_state = getattr(buffer, "get_state", None)
            if callable(get_state):
                states[name] = get_state()
        return states

    def _restore_buffer_states(self, buffers: Mapping[str, Any], states: Mapping[str, Any]) -> None:
        for name, buffer in buffers.items():
            if name not in states:
                raise RuntimeError(
                    f"Exact resume requires checkpoint buffer state for '{name}'. "
                    "Use warm_start_optimizer for research continuation when replay state is unavailable."
                )
            set_state = getattr(buffer, "set_state", None)
            if not callable(set_state):
                raise RuntimeError(
                    f"Exact resume requires buffer '{name}' to implement set_state(). "
                    "Use warm_start_optimizer for research continuation when replay state is unavailable."
                )
            set_state(states[name])

    def load_checkpoint(self, path: Path, *, mode: str = "exact") -> None:
        """Load checkpoint and restore state according to the requested mode.

        Modes:
            exact: restore training state for fault-tolerant continuation.
            warm_start: restore learned weights/controllers, use new schedule and optimizer.
            warm_start_optimizer: restore learned weights/controllers/optimizers, use new schedule.
        """
        valid_modes = {"exact", "warm_start", "warm_start_optimizer"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown resume mode '{mode}'. Expected one of {sorted(valid_modes)}.")

        # 1. Load raw checkpoint
        ckpt = self.checkpoint_manager.load(path)
        if ckpt is None:
            logger.warning(f"No checkpoint found at {path}")
            return

        context = self.ensure_context()

        exact_resume = mode == "exact"
        load_optimizer_state = mode in {"exact", "warm_start_optimizer"}

        # 2. Restore or reset global step
        checkpoint_step = ckpt.get("global_step", 0)
        if exact_resume:
            self.global_step = checkpoint_step
            self.best_eval_return = float(ckpt.get("best_eval_return", float("-inf")))
        else:
            self.global_step = 0
            self.best_eval_return = float("-inf")
        logger.info(
            "Restoring checkpoint from step %s with mode '%s' (run step now %s)",
            checkpoint_step,
            mode,
            self.global_step,
        )

        # 3. Restore phase scheduler state
        if exact_resume:
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
        if load_optimizer_state:
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
        if exact_resume:
            workflow_custom = ckpt.get("workflow_custom", {})
            self.workflow.set_state(workflow_custom)
            logger.debug("Restored workflow custom state")

        # 8. Restore buffers for fault-tolerant continuation.
        if exact_resume:
            self._restore_buffer_states(context.buffers, ckpt.get("buffers", {}))
            logger.debug("Restored buffer states")

        # 9. Update context global_step
        self._update_context(global_step=self.global_step)

        # 10. Restore RNG states
        if exact_resume:
            self.checkpoint_manager.restore_rng_states(ckpt)

        logger.info("Checkpoint restored successfully with mode '%s'", mode)

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
