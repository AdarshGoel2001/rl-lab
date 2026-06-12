from pathlib import Path

import os
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.environments.dmc_wrapper import DMCWrapper
from src.workflows.dreamer import DreamerV1Workflow


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_dmc_pixel_observation_stays_channel_first_image():
    wrapper = DMCWrapper.__new__(DMCWrapper)
    wrapper.from_pixels = True

    obs = wrapper._flatten_obs({"pixels": np.zeros((64, 64, 3), dtype=np.uint8)})

    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.uint8


def test_dmc_pixel_mode_defaults_to_egl_before_dm_control_import(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    wrapper = DMCWrapper.__new__(DMCWrapper)
    wrapper.from_pixels = True
    wrapper.config = {}

    wrapper._configure_pixel_render_backend()

    assert os.environ["MUJOCO_GL"] == "egl"


def test_dreamer_image_encoder_decoder_round_trip_shapes():
    from src.components.encoders.dreamer_cnn import DreamerImageEncoder
    from src.components.decoders.dreamer_cnn import DreamerImageDecoder

    encoder = DreamerImageEncoder(input_shape=(3, 64, 64), output_dim=128, depth=8)
    decoder = DreamerImageDecoder(input_dim=80, output_shape=(3, 64, 64), depth=8)

    images = torch.randint(0, 256, (4, 6, 3, 64, 64), dtype=torch.uint8)
    embeddings = encoder(images)
    recon = decoder(torch.randn(4, 6, 80))

    assert embeddings.shape == (4, 6, 128)
    assert embeddings.dtype == torch.float32
    assert recon.shape == (4, 6, 3, 64, 64)
    assert recon.dtype == torch.float32


def test_dreamer_pixel_world_model_update_encodes_images_and_reconstructs_pixels():
    from src.components.decoders.dreamer_cnn import DreamerImageDecoder
    from src.components.encoders.dreamer_cnn import DreamerImageEncoder
    from src.components.prediction_heads.mlp import MLPHead
    from src.components.representation_learners.rssm import RSSMRepresentationLearner

    encoder = DreamerImageEncoder(input_shape=(3, 64, 64), output_dim=32, depth=4)
    rssm = RSSMRepresentationLearner(
        feature_dim=32,
        action_dim=1,
        deterministic_dim=16,
        stochastic_dim=8,
        hidden_dim=32,
        min_std=0.1,
        device="cpu",
    )
    latent_dim = rssm.representation_dim
    decoder = DreamerImageDecoder(input_dim=latent_dim, output_shape=(3, 64, 64), depth=4)
    reward = MLPHead(input_dim=latent_dim, output_dim=1, hidden_dim=32)
    continuation = MLPHead(input_dim=latent_dim, output_dim=1, hidden_dim=32)

    workflow = DreamerV1Workflow()
    workflow.device = "cpu"
    workflow.observation_encoder = encoder
    workflow.rssm = rssm
    workflow.reward_predictor = reward
    workflow.continue_predictor = continuation
    workflow.observation_predictor = decoder
    workflow.world_model_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(reward.parameters())
        + list(continuation.parameters()),
        lr=1e-3,
    )
    workflow.free_nats = 0.0
    workflow.kl_scale = 1.0
    workflow.observation_loss_scale = 1.0

    batch = {
        "observations": torch.randint(0, 256, (2, 5, 3, 64, 64), dtype=torch.uint8),
        "actions": torch.zeros(2, 5, 1),
        "rewards": torch.randn(2, 5),
        "dones": torch.zeros(2, 5, dtype=torch.bool),
    }

    metrics = workflow.update_world_model(batch, phase={"name": "train_world_model"})

    assert metrics["world_model/total_loss"] > 0.0
    assert metrics["world_model/observation_loss"] > 0.0
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_dreamer_pixel_tiny_config_resolves():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=dreamer_dmc_cartpole_swingup", "budget=dreamer_pixel_tiny"],
        )

    assert cfg.environment.from_pixels is True
    assert tuple(cfg.environment.height for _ in [0]) == (64,)
    assert cfg._dims.encoder_output == 128
    assert cfg.components.observation_encoder._target_ == "src.components.encoders.dreamer_cnn.DreamerImageEncoder"
    assert cfg.components.observation_predictor._target_ == "src.components.decoders.dreamer_cnn.DreamerImageDecoder"
    assert cfg.components.representation_learner.feature_dim == cfg._dims.encoder_output
    assert cfg.components.observation_predictor.output_shape == [3, 64, 64]


def test_dreamer_pixel_100ep_config_uses_larger_visual_model():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=dreamer_dmc_cartpole_swingup", "budget=dreamer_pixel_100ep"],
        )

    assert cfg.environment.from_pixels is True
    assert cfg.environment.height == 64
    assert cfg.environment.width == 64
    assert cfg._dims.encoder_output == 1024
    assert cfg.components.observation_encoder.depth == 32
    assert cfg.components.observation_predictor.depth == 32
    assert cfg._dims.stochastic == 30
    assert cfg._dims.deterministic == 200
    assert cfg.training.num_eval_episodes == 5
    assert cfg.training.eval_frequency == 4096
