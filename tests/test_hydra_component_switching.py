from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose_component(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def test_representation_learner_configs_are_swappable_at_component_level():
    identity_cfg = _compose_component(
        "components/representation_learner/identity",
        ["+_dims.representation=8"],
    )
    rssm_cfg = _compose_component(
        "components/representation_learner/rssm",
        [
            "+_dims.encoder_output=8",
            "+components.representation_learner.action_dim=0",
            "components.representation_learner.deterministic_dim=16",
            "components.representation_learner.stochastic_dim=4",
            "components.representation_learner.hidden_dim=16",
        ],
    )

    identity = instantiate(identity_cfg.components.representation_learner)
    rssm = instantiate(rssm_cfg.components.representation_learner)

    features = torch.randn(2, 3, 8)
    dones = torch.zeros(2, 3, dtype=torch.bool)

    identity_sequence = identity.observe_sequence(features, dones=dones)
    rssm_sequence = rssm.observe_sequence(features, dones=dones)

    assert identity_sequence.posterior.to_tensor().shape == (2, 3, 16)
    assert rssm_sequence.posterior.to_tensor().shape == (2, 3, 20)
