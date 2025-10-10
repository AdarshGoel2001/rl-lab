"""World-model paradigm that delegates to the modular world model system."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch.distributions import Distribution

from ..base import BaseParadigm
from ...components.world_models.dynamics.base import BaseDynamicsModel
from ...components.encoders.base import BaseEncoder
from ...components.world_models.controllers.base import BaseController
from ...components.policy_heads.base import BasePolicyHead
from ...components.world_models.representation_learners import BaseRepresentationLearner
from ...components.value_functions.base import BaseValueFunction
from ...components.world_models.latents import LatentBatch
from ...components.world_models.predictors import BaseRewardPredictor
from ...components.world_models.decoders import BaseObservationDecoder
from ...utils.registry import register_paradigm
from .system import WorldModelSystem


@register_paradigm("world_model")
class WorldModelParadigm(BaseParadigm):
    """High-level RL paradigm that learns and leverages world models."""

    def __init__(
        self,
        *,
        encoder: BaseEncoder,
        representation_learner: BaseRepresentationLearner,
        dynamics_model: BaseDynamicsModel,
        policy_head: BasePolicyHead,
        value_function: BaseValueFunction,
        reward_predictor: Optional[BaseRewardPredictor] = None,
        planner: Optional[BaseController] = None,
        observation_decoder: Optional[BaseObservationDecoder] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(encoder, representation_learner, policy_head, config)

        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.reward_predictor = reward_predictor
        self.planner = planner
        self.observation_decoder = observation_decoder

        self.dynamics_model.to(self.device)
        self.value_function.to(self.device)
        if self.reward_predictor is not None:
            self.reward_predictor.to(self.device)
        if self.planner is not None:
            self.planner.to(self.device)
        if self.observation_decoder is not None:
            self.observation_decoder.to(self.device)

        self.system = WorldModelSystem(
            encoder=self.encoder,
            representation_learner=self.representation_learner,
            dynamics_model=self.dynamics_model,
            policy_head=self.policy_head,
            value_function=self.value_function,
            reward_predictor=self.reward_predictor,
            planner=self.planner,
            observation_decoder=self.observation_decoder,
            config=self.config,
            device=self.device,
        )

    def forward(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Distribution:
        return self.system.act(observations, context=context)

    def get_value(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        return self.system.value(observations)

    def rollout_imagination(
        self,
        initial_states: torch.Tensor,
        length: int,
    ) -> Dict[str, torch.Tensor]:
        latent_batch = LatentBatch(latent=initial_states, features=None)
        return self.system.imagine(latent_batch, horizon=length)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.system.compute_losses(batch)

    def to(self, device: Union[str, torch.device]) -> "WorldModelParadigm":
        super().to(device)
        target_device = torch.device(device)
        if hasattr(self, "system") and self.system is not None:
            self.system.to(target_device)
        if self.config is not None:
            self.config['device'] = str(target_device)
        return self

    def _save_additional_components(self) -> Dict[str, Any]:
        additional = {
            "dynamics_model": self.dynamics_model.state_dict(),
            "value_function": self.value_function.state_dict(),
        }
        if self.reward_predictor is not None:
            additional["reward_predictor"] = self.reward_predictor.state_dict()
        if self.planner is not None:
            additional["planner"] = self.planner.state_dict()
        if self.observation_decoder is not None:
            additional["observation_decoder"] = self.observation_decoder.state_dict()
        return additional

    def _load_additional_components(self, checkpoint: Dict[str, Any]) -> None:
        if "dynamics_model" in checkpoint:
            self.dynamics_model.load_state_dict(checkpoint["dynamics_model"])
        if "value_function" in checkpoint:
            self.value_function.load_state_dict(checkpoint["value_function"])
        if "reward_predictor" in checkpoint and self.reward_predictor is not None:
            self.reward_predictor.load_state_dict(checkpoint["reward_predictor"])
        if "planner" in checkpoint and self.planner is not None:
            self.planner.load_state_dict(checkpoint["planner"])
        if "observation_decoder" in checkpoint and self.observation_decoder is not None:
            self.observation_decoder.load_state_dict(checkpoint["observation_decoder"])

    def get_paradigm_info(self) -> Dict[str, Any]:
        info = super().get_paradigm_info()
        info["dynamics_info"] = self.dynamics_model.get_dynamics_info()
        info["value_info"] = self.value_function.get_value_info()
        if self.reward_predictor is not None:
            info["reward_predictor"] = {
                "type": self.reward_predictor.__class__.__name__,
            }
        if self.planner is not None:
            info["planner_info"] = self.planner.get_planner_info()
        if self.observation_decoder is not None:
            info["observation_decoder"] = {
                "type": self.observation_decoder.__class__.__name__,
            }
        return info
