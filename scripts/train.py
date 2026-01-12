#!/usr/bin/env python3
"""
Hydra-powered training entry point for world-model experiments.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Ensure local packages are importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration import Orchestrator
from src.utils.config import resolve_device
from src.workflows import DreamerWorkflow
from src.workflows.utils.context import ControllerManager, WorldModelComponents

logger = logging.getLogger(__name__)

if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------



def set_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]




# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_world_model_components(cfg: DictConfig, device: str) -> WorldModelComponents:
    """Instantiate all declared world-model components via Hydra."""
    comp_cfg = cfg.components
    components: Dict[str, Any] = {}

    for name, component_cfg in comp_cfg.items():
        if component_cfg is None:
            continue
        try:
            component = instantiate(component_cfg, device=device)
        except TypeError:
            component = instantiate(component_cfg)
        components[name] = component

    return WorldModelComponents(
        components=components,
        config=config_to_dict(comp_cfg),
    ).to(device)


def build_optimizers(
    cfg: DictConfig,
    components: WorldModelComponents,
    controllers: Dict[str, Any],
) -> Dict[str, torch.optim.Optimizer]:
    """Build optimizers based on config, falling back to legacy defaults."""
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    optimizer_cfg = getattr(cfg, "optimizers", None)

    def _world_model_params() -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for module in components.components.values():
            if isinstance(module, torch.nn.Module):
                params.extend(list(module.parameters()))
        return params

    if optimizer_cfg:
        if "world_model" in optimizer_cfg:
            params = _world_model_params()
            if not params:
                raise RuntimeError("Configured world_model optimizer but no module parameters were found.")
            factory = instantiate(optimizer_cfg.world_model)
            optimizer = factory(params) if callable(factory) else factory  # type: ignore[call-arg]
            if callable(factory) is False and getattr(optimizer, "param_groups", None):
                if not optimizer.param_groups:
                    logger.warning("World-model optimizer has no parameter groups; verify configuration.")
            optimizers["world_model"] = optimizer
        for role, controller in controllers.items():
            if role not in optimizer_cfg:
                continue
            factory = instantiate(optimizer_cfg[role])
            params = list(controller.parameters()) if hasattr(controller, "parameters") else []
            optimizer = factory(params) if callable(factory) else factory  # type: ignore[call-arg]
            if callable(factory) is False and getattr(optimizer, "param_groups", None) and not optimizer.param_groups:
                logger.warning("Optimizer for controller '%s' has no parameters; verify configuration.", role)
            optimizers[role] = optimizer
            if hasattr(controller, "optimizer"):
                controller.optimizer = optimizer
    else:
        params = _world_model_params()
        if not params:
            raise RuntimeError("No parameters available to construct world-model optimizer.")
        lr = float(cfg.algorithm.get("world_model_lr", 2e-4))
        betas = cfg.algorithm.get("world_model_betas", (0.9, 0.999))
        weight_decay = float(cfg.algorithm.get("world_model_weight_decay", 0.0))
        optimizers["world_model"] = torch.optim.Adam(params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)

    return optimizers


def build_controllers(cfg: DictConfig, device: str) -> Tuple[Dict[str, Any], ControllerManager]:
    controllers: Dict[str, Any] = {}
    for role, controller_cfg in cfg.controllers.items():
        try:
            controller = instantiate(controller_cfg, device=device)
        except TypeError:
            controller = instantiate(controller_cfg)
        if hasattr(controller, "to"):
            controller.to(device)
        controllers[role] = controller
    return controllers, ControllerManager(controllers)


# ---------------------------------------------------------------------------
# Hydra main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Working directory: %s", os.getcwd())
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.experiment.device)

    set_seeds(cfg.experiment.get("seed", None))

    workflow: DreamerWorkflow = instantiate(cfg.workflow)
    components = build_world_model_components(cfg, device)
    controllers, controller_manager = build_controllers(cfg, device)
    optimizers = build_optimizers(cfg, components, controllers)

    train_environment = instantiate(cfg.environment)
    eval_environment = instantiate(cfg.evaluation) if "evaluation" in cfg and cfg.evaluation is not None else None

    buffer_num_envs = getattr(train_environment, "num_envs", None)
    if buffer_num_envs is None:
        buffer_num_envs = cfg.environment.get("num_envs", 1)

    buffers: Dict[str, Any] = {}
    if "buffers" in cfg and cfg.buffers is not None:
        for name, buffer_cfg in cfg.buffers.items():
            try:
                buffers[name] = instantiate(buffer_cfg, device=device, num_envs=buffer_num_envs)
            except TypeError:
                try:
                    buffers[name] = instantiate(buffer_cfg, device=device)
                except TypeError:
                    buffers[name] = instantiate(buffer_cfg)

    if not buffers:
        buffers["replay"] = instantiate(cfg.buffer, device=device, num_envs=buffer_num_envs)

    orchestrator = Orchestrator(
        cfg,
        workflow,
        components=components,
        optimizers=optimizers,
        controllers=controllers,
        controller_manager=controller_manager,
        buffers=buffers,
        train_environment=train_environment,
        eval_environment=eval_environment,
    )

    # Handle checkpoint resume if specified
    resume_path = cfg.training.get("resume_path", None)
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            orchestrator.load_checkpoint(resume_path)
        else:
            logger.warning(f"Resume path not found, starting fresh: {resume_path}")

    logger.info("Starting training loop...")
    results = orchestrator.train()
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
