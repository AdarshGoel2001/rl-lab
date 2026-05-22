from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.validate_experiment import validate_experiment_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_planet_dmc_cartpole_swingup_config_resolves_to_dmc_state_setup():
    result = validate_experiment_config("planet_dmc_cartpole_swingup")

    assert result["ok"] is True
    assert result["targets"]["environment"] == "src.environments.dmc_wrapper.DMCWrapper"
    assert result["targets"]["components.observation_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["phases"] == [
        "collect_random_seed",
        "train_world_model_seed",
        "collect_with_planner_1",
        "train_world_model_1",
        "collect_with_planner_2",
        "train_world_model_2",
        "final_eval",
    ]


def test_planet_dmc_cartpole_swingup_dimensions_and_mac_safe_limits():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=planet_dmc_cartpole_swingup"])

    assert cfg.environment.name == "cartpole_swingup"
    assert cfg.environment.from_pixels is False
    assert cfg.environment.frame_skip == 8
    assert cfg.environment.max_episode_steps == 125

    assert cfg._dims.observation == 5
    assert cfg._dims.action == 1
    assert cfg._dims.stochastic == 16
    assert cfg._dims.deterministic == 64
    assert cfg._dims.hidden == 128
    assert cfg._dims.representation == 80

    assert cfg.buffer.batch_size <= 16
    assert cfg.buffer.sequence_length in (16, 25)
    assert cfg.algorithm.free_nats == 3.0
    assert cfg.algorithm.max_grad_norm == 1000.0
    assert cfg.algorithm.world_model_lr == 1.0e-3

    assert cfg.controllers.actor.action_dim == 1
    assert cfg.controllers.actor.action_low == [-1.0]
    assert cfg.controllers.actor.action_high == [1.0]
    assert cfg.controllers.planner.action_dim == 1
    assert cfg.controllers.planner.horizon in (8, 12)
    assert cfg.controllers.planner.num_samples <= 64
    assert cfg.controllers.planner.iterations <= 3

    assert cfg.components.observation_predictor.input_dim == cfg._dims.representation
    assert cfg.components.observation_predictor.output_dim == cfg._dims.observation
