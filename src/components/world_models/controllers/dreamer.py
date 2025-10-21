"""Dreamer-specific actor and critic controllers.

These controllers encapsulate the policy/value heads and their optimizers so
that Dreamer workflows can delegate action selection and gradient updates
through the controller manager rather than relying on standalone components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution, Independent, Normal

from ....utils.registry import register_controller
from .base import BaseController


def _build_activation(name: str) -> nn.Module:
    """Resolve activation name to module."""
    name = (name or "elu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "swish" or name == "silu":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name == "elu":
        return nn.ELU()
    return nn.ReLU()


def _flatten_time(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten (B, T, ...) tensors to (B*T, ...)."""
    if tensor.dim() <= 1:
        return tensor
    batch, horizon = tensor.shape[:2]
    remaining = tensor.shape[2:]
    if remaining:
        return tensor.reshape(batch * horizon, *remaining)
    return tensor.reshape(batch * horizon)


def _ensure_tensor(value: Any, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert arbitrary tensor-like input to a torch.Tensor on the requested device."""
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor
    tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


@dataclass
class _ControllerSpecs:
    representation_dim: int
    action_dim: int
    discrete_actions: bool


def _infer_specs(config: Dict[str, Any], components: Optional[Any]) -> _ControllerSpecs:
    """Infer latent/action specifications from controller config or component bundle."""
    rep_dim = config.get("representation_dim")
    action_dim = config.get("action_dim")
    discrete = bool(config.get("discrete_actions", False))

    if components is not None:
        rep = getattr(components, "representation_learner", None)
        specs_dict = getattr(components, "specs", {}) or {}
        if rep_dim is None:
            for attr in ("representation_dim", "feature_dim", "latent_dim"):
                rep_dim = getattr(rep, attr, None)
                if rep_dim is not None:
                    break
            if rep_dim is None:
                rep_dim = specs_dict.get("representation_dim")
        if action_dim is None:
            action_dim = specs_dict.get("action_dim")
            if action_dim is None:
                config_space = getattr(components, "config", {}) or {}
                action_dim = config_space.get("action_dim")
        if "discrete_actions" not in config:
            if "discrete_actions" in specs_dict:
                discrete = bool(specs_dict["discrete_actions"])
            else:
                policy = getattr(components, "policy_head", None)
                if policy is not None and hasattr(policy, "discrete_actions"):
                    discrete = bool(getattr(policy, "discrete_actions"))

    if rep_dim is None:
        raise ValueError("Dreamer controller requires 'representation_dim' in config or component bundle.")
    if action_dim is None:
        raise ValueError("Dreamer controller requires 'action_dim' in config or component bundle.")

    return _ControllerSpecs(
        representation_dim=int(rep_dim),
        action_dim=int(action_dim),
        discrete_actions=discrete,
    )


class _ActorBackbone(nn.Module):
    """Simple MLP backbone shared between actor/critic implementations."""

    def __init__(self, *, input_dim: int, hidden_dims: Tuple[int, ...], activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_build_activation(activation))
            current_dim = hidden_dim

        self.model = nn.Sequential(*layers)
        self.output_dim = current_dim
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(self.model) == 0:
            return inputs
        return self.model(inputs)


@register_controller("dreamer_actor")
class DreamerActorController(BaseController):
    """Dreamer policy controller that owns the action head and optimizer."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        components = cfg.pop("components", None)
        self.role = cfg.get("role", "actor")

        super().__init__()
        self.device = torch.device(cfg.get("device", "cpu"))
        specs = _infer_specs(cfg, components)
        self.representation_dim = specs.representation_dim
        self.action_dim = specs.action_dim
        self.discrete_actions = specs.discrete_actions

        hidden_dims = tuple(cfg.get("hidden_dims", (256, 256)))
        activation = cfg.get("activation", "elu")
        self.entropy_coef = float(cfg.get("entropy_coef", 0.0))
        self.grad_clip = cfg.get("grad_clip")
        self.normalize_advantage = bool(cfg.get("normalize_advantage", True))
        self.return_key = cfg.get("return_key", "returns")
        self.advantage_key = cfg.get("advantage_key", "advantages")
        self.discount_key = cfg.get("discount_key", "discounts")
        self.weight_key = cfg.get("weights")

        self.min_std = float(cfg.get("min_std", 0.1))
        self.max_std = cfg.get("max_std")
        self.init_std = float(cfg.get("init_std", 0.0))
        self.std_transform = cfg.get("std_transform", "softplus")
        self.reparameterized = bool(cfg.get("reparameterized", True))

        self.backbone = _ActorBackbone(
            input_dim=self.representation_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        if self.discrete_actions:
            self.policy_head = nn.Linear(self.backbone.output_dim, self.action_dim)
            nn.init.orthogonal_(self.policy_head.weight)
            nn.init.constant_(self.policy_head.bias, 0.0)
        else:
            self.policy_mean = nn.Linear(self.backbone.output_dim, self.action_dim)
            self.policy_std = nn.Linear(self.backbone.output_dim, self.action_dim)
            nn.init.orthogonal_(self.policy_mean.weight)
            nn.init.constant_(self.policy_mean.bias, 0.0)
            nn.init.orthogonal_(self.policy_std.weight)
            nn.init.constant_(self.policy_std.bias, 0.0)

        lr = float(cfg.get("lr", cfg.get("learning_rate", 3e-4)))
        betas = cfg.get("betas", (0.9, 0.999))
        weight_decay = float(cfg.get("weight_decay", 0.0))

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

        self.to(self.device)

    # ------------------------------------------------------------------
    # Acting API
    # ------------------------------------------------------------------
    def forward(self, latent_state: torch.Tensor) -> Distribution:
        latent_state = latent_state.to(self.device)
        hidden = self.backbone(latent_state)

        if self.discrete_actions:
            logits = self.policy_head(hidden)
            return Categorical(logits=logits)

        mean = self.policy_mean(hidden)
        std_param = self.policy_std(hidden)
        if self.std_transform == "softplus":
            std = F.softplus(std_param + self.init_std)
        elif self.std_transform == "exp":
            std = torch.exp(std_param + self.init_std)
        else:
            std = F.softplus(std_param + self.init_std)

        std = std + self.min_std
        if self.max_std is not None:
            std = torch.clamp(std, max=float(self.max_std))

        base_dist = Normal(mean, std)
        # Ensure vector actions return correct batch shape
        return Independent(base_dist, 1)

    def act(  # type: ignore[override]
        self,
        latent_state: torch.Tensor,
        dynamics_model: Optional[Any] = None,
        value_function: Optional[Any] = None,
        *,
        deterministic: bool = False,
        horizon: Optional[int] = None,
        **_: Any,
    ) -> Distribution:
        _ = dynamics_model, value_function, horizon  # Unused for Dreamer actor
        return self.forward(latent_state)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(
        self,
        batch: Mapping[str, Any],
        *,
        phase: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        del phase  # Controller currently does not use phase metadata.

        if "states" not in batch:
            raise KeyError("Dreamer actor controller expects 'states' in learning batch.")
        if "actions" not in batch:
            raise KeyError("Dreamer actor controller expects 'actions' in learning batch.")

        states = _ensure_tensor(batch["states"], device=self.device, dtype=torch.float32)
        actions = _ensure_tensor(batch["actions"], device=self.device)

        states = _flatten_time(states)
        actions = _flatten_time(actions)

        distribution = self.forward(states)

        log_probs = self._compute_log_prob(distribution, actions)

        advantages = batch.get(self.advantage_key)
        returns = batch.get(self.return_key)
        values = batch.get("values")

        advantage_tensor = None
        if advantages is not None:
            advantage_tensor = _ensure_tensor(advantages, device=self.device, dtype=torch.float32)
        elif returns is not None and values is not None:
            returns_tensor = _ensure_tensor(returns, device=self.device, dtype=torch.float32)
            values_tensor = _ensure_tensor(values, device=self.device, dtype=torch.float32)
            advantage_tensor = returns_tensor - values_tensor
        elif returns is not None:
            advantage_tensor = _ensure_tensor(returns, device=self.device, dtype=torch.float32)

        if advantage_tensor is None:
            raise KeyError(
                "Dreamer actor controller requires either 'advantages' or 'returns' in the batch."
            )

        advantage_tensor = _flatten_time(advantage_tensor)

        weights = None
        if self.weight_key and self.weight_key in batch:
            weights = _ensure_tensor(batch[self.weight_key], device=self.device, dtype=torch.float32)
            weights = _flatten_time(weights)
        elif self.discount_key in batch:
            weights = _ensure_tensor(batch[self.discount_key], device=self.device, dtype=torch.float32)
            weights = _flatten_time(weights)

        if weights is not None:
            weights = weights.detach()
            mask = weights
        else:
            mask = None

        advantage_tensor = advantage_tensor.detach()
        if self.normalize_advantage and advantage_tensor.numel() > 1:
            eps = 1e-6
            adv_mean = advantage_tensor.mean()
            adv_std = advantage_tensor.std(unbiased=False)
            advantage_tensor = (advantage_tensor - adv_mean) / (adv_std + eps)

        if mask is not None:
            advantage_tensor = advantage_tensor * mask
            log_probs = log_probs * mask

        entropy = self._compute_entropy(distribution, mask)

        denom = mask.sum() if mask is not None else torch.tensor(log_probs.shape[0], device=self.device, dtype=torch.float32)
        denom = torch.clamp(denom, min=1.0)

        policy_loss = -(advantage_tensor * log_probs).sum() / denom
        entropy_bonus = self.entropy_coef * entropy
        loss = policy_loss - entropy_bonus

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip))
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "adv_mean": float(advantage_tensor.mean().item()),
            "adv_std": float(advantage_tensor.std(unbiased=False).item()),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _compute_log_prob(self, distribution: Distribution, actions: torch.Tensor) -> torch.Tensor:
        if self.discrete_actions:
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            if actions.dtype != torch.long:
                if actions.dim() > 1 and actions.shape[-1] == self.action_dim:
                    actions = actions.argmax(dim=-1)
                else:
                    actions = actions.to(torch.long)
            log_prob = distribution.log_prob(actions)
        else:
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
            log_prob = distribution.log_prob(actions.to(torch.float32))
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def _compute_entropy(self, distribution: Distribution, mask: Optional[torch.Tensor]) -> torch.Tensor:
        entropy = distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        if mask is not None:
            entropy = (entropy * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        else:
            entropy = entropy.mean()
        return entropy

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        model_state = super().state_dict(destination, prefix, keep_vars)
        return {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "metadata": {
                "representation_dim": self.representation_dim,
                "action_dim": self.action_dim,
                "discrete_actions": self.discrete_actions,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):  # type: ignore[override]
        if "model" in state_dict:
            super().load_state_dict(state_dict["model"], strict=strict)
            if "optimizer" in state_dict and state_dict["optimizer"]:
                self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            super().load_state_dict(state_dict, strict=strict)


@register_controller("dreamer_critic")
class DreamerCriticController(BaseController):
    """Dreamer critic that predicts value estimates from latent states."""

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = dict(config)
        components = cfg.pop("components", None)
        self.role = cfg.get("role", "critic")

        super().__init__()
        self.device = torch.device(cfg.get("device", "cpu"))
        specs = _infer_specs(cfg, components)
        self.representation_dim = specs.representation_dim

        hidden_dims = tuple(cfg.get("hidden_dims", (512, 512)))
        activation = cfg.get("activation", "elu")
        self.grad_clip = cfg.get("grad_clip")
        self.target_key = cfg.get("target_key", "returns")
        self.discount_key = cfg.get("discount_key", "discounts")
        self.weight_key = cfg.get("weights")
        self.huber_delta = cfg.get("huber_delta")

        self.backbone = _ActorBackbone(
            input_dim=self.representation_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.value_head = nn.Linear(self.backbone.output_dim, 1)
        nn.init.orthogonal_(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0.0)

        lr = float(cfg.get("lr", cfg.get("learning_rate", 3e-4)))
        betas = cfg.get("betas", (0.9, 0.999))
        weight_decay = float(cfg.get("weight_decay", 0.0))

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )

        self.to(self.device)

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        latent_state = latent_state.to(self.device)
        hidden = self.backbone(latent_state)
        value = self.value_head(hidden)
        return value.squeeze(-1)

    def act(  # type: ignore[override]
        self,
        latent_state: torch.Tensor,
        dynamics_model: Optional[Any] = None,
        value_function: Optional[Any] = None,
        *,
        deterministic: bool = False,
        horizon: Optional[int] = None,
        **kwargs: Any,
    ) -> Distribution:
        raise NotImplementedError("Dreamer critic does not implement act().")

    def learn(
        self,
        batch: Mapping[str, Any],
        *,
        phase: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        del phase  # Phase metadata currently unused.

        if "states" not in batch:
            raise KeyError("Dreamer critic controller expects 'states' in learning batch.")
        if self.target_key not in batch:
            raise KeyError(f"Dreamer critic controller expects '{self.target_key}' in learning batch.")

        states = _ensure_tensor(batch["states"], device=self.device, dtype=torch.float32)
        targets = _ensure_tensor(batch[self.target_key], device=self.device, dtype=torch.float32)

        states = _flatten_time(states)
        targets = _flatten_time(targets)

        predictions = self.forward(states)
        if self.huber_delta is not None:
            delta = float(self.huber_delta)
            loss_vec = F.huber_loss(predictions, targets, delta=delta, reduction="none")
        else:
            loss_vec = F.mse_loss(predictions, targets, reduction="none")

        weights = None
        if self.weight_key and self.weight_key in batch:
            weights = _ensure_tensor(batch[self.weight_key], device=self.device, dtype=torch.float32)
            weights = _flatten_time(weights)
        elif self.discount_key in batch:
            weights = _ensure_tensor(batch[self.discount_key], device=self.device, dtype=torch.float32)
            weights = _flatten_time(weights)

        if weights is not None:
            weighted_loss = loss_vec * weights
            denom = torch.clamp(weights.sum(), min=1.0)
            value_loss = weighted_loss.sum() / denom
        else:
            value_loss = loss_vec.mean()

        self.optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), float(self.grad_clip))
        self.optimizer.step()

        with torch.no_grad():
            value_mean = predictions.mean().item()
            target_mean = targets.mean().item()

        return {
            "loss": float(value_loss.item()),
            "value_mean": float(value_mean),
            "target_mean": float(target_mean),
            "mse": float(loss_vec.mean().item()),
        }

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        model_state = super().state_dict(destination, prefix, keep_vars)
        return {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "metadata": {
                "representation_dim": self.representation_dim,
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):  # type: ignore[override]
        if "model" in state_dict:
            super().load_state_dict(state_dict["model"], strict=strict)
            if "optimizer" in state_dict and state_dict["optimizer"]:
                self.optimizer.load_state_dict(state_dict["optimizer"])
        else:
            super().load_state_dict(state_dict, strict=strict)
