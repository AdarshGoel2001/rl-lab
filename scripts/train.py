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

from src.orchestration import WorldModelOrchestrator
from src.utils.config import resolve_device
from src.workflows.world_models import DreamerWorkflow
from src.workflows.world_models.context import ControllerManager, WorldModelComponents

logger = logging.getLogger(__name__)

if not OmegaConf.has_resolver("add"):
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def validate_config(cfg: DictConfig) -> None:
    """Fail fast if mandatory sections or dimensions are missing."""
    required_sections = ["experiment", "environment", "components", "controllers", "workflow", "training", "algorithm"]
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    if "_dims" not in cfg:
        raise ValueError("Config is missing '_dims' section describing core dimensions.")

    dims = cfg._dims
    comp = cfg.components

    encoder_out = comp.encoder.get("output_dim", dims.get("encoder_output"))
    if encoder_out is None:
        raise ValueError("components.encoder.output_dim or _dims.encoder_output must be defined.")

    rep_feat = comp.representation_learner.get("feature_dim")
    if rep_feat is not None and rep_feat != encoder_out:
        raise ValueError(
            f"Dimension mismatch: encoder output {encoder_out} != representation_learner.feature_dim {rep_feat}."
        )

    det_dim = comp.representation_learner.get("deterministic_dim", 0)
    stoch_dim = comp.representation_learner.get("stochastic_dim", 0)
    expected_rep = det_dim + stoch_dim
    rep_dim = dims.get("representation")
    if rep_dim is not None and rep_dim != expected_rep:
        raise ValueError(
            f"Dimension mismatch: _dims.representation ({rep_dim}) != deterministic+stochastic ({expected_rep})."
        )

    action_dim = dims.get("action")
    if action_dim is None:
        raise ValueError("Config must define _dims.action for controller wiring.")

    actor_cfg = cfg.controllers.actor
    if actor_cfg.get("action_dim") not in (None, action_dim):
        raise ValueError(
            f"Controller action_dim mismatch: controller={actor_cfg.get('action_dim')} expected={action_dim}."
        )


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


def validate_instantiated_components(cfg: DictConfig, components: WorldModelComponents, controllers: Dict[str, Any]) -> None:
    """Validate actual component dimensions match config expectations."""
    dims = cfg._dims

    expected_encoder = dims.get("encoder_output")
    encoder = getattr(components, "encoder", None)
    actual_encoder = getattr(encoder, "output_dim", None) if encoder is not None else None
    if expected_encoder is not None and actual_encoder is not None and actual_encoder != expected_encoder:
        raise ValueError(
            f"Encoder output_dim {actual_encoder} does not match _dims.encoder_output {expected_encoder}."
        )

    expected_rep = dims.get("representation")
    representation = getattr(components, "representation_learner", None)
    actual_rep = getattr(representation, "representation_dim", None) if representation is not None else None
    if expected_rep is not None and actual_rep is not None and actual_rep != expected_rep:
        raise ValueError(
            f"Representation learner dimension {actual_rep} does not match _dims.representation {expected_rep}."
        )

    expected_action = dims.get("action")
    actor = controllers.get("actor") if controllers else None
    actual_action = getattr(actor, "action_dim", None) if actor is not None else None
    if expected_action is not None and actual_action is not None and actual_action != expected_action:
        raise ValueError(
            f"Actor action_dim {actual_action} does not match _dims.action {expected_action}."
        )


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

    validate_config(cfg)

    logger.info("Working directory: %s", os.getcwd())
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.experiment.device)

    set_seeds(cfg.experiment.get("seed", None))

    workflow: DreamerWorkflow = instantiate(cfg.workflow)
    components = build_world_model_components(cfg, device)
    controllers, controller_manager = build_controllers(cfg, device)
    optimizers = build_optimizers(cfg, components, controllers)
    validate_instantiated_components(cfg, components, controllers)

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

    orchestrator = WorldModelOrchestrator(
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

    logger.info("Starting training loop...")
    results = orchestrator.train()
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
