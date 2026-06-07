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


def test_planet_dmc_chapter_budget_is_real_eval_attempt():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_chapter")

    assert result["ok"] is True
    assert result["phases"] == [
        "collect_random_seed",
        "train_world_model_seed",
        "collect_with_planner_1",
        "train_world_model_1",
        "collect_with_planner_2",
        "train_world_model_2",
        "collect_with_planner_3",
        "train_world_model_3",
        "final_eval",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_chapter"],
        )

    assert cfg.experiment.name == "planet_dmc_swingup_chapter"
    assert cfg.environment.name == "cartpole_swingup"
    assert cfg.training.num_eval_episodes == 5
    assert cfg.training.max_eval_steps == 125
    assert cfg.training.phases[0].duration_steps >= 1024
    assert cfg.training.phases[1].duration_updates >= 256
    assert cfg.training.phases[-1].workflow_hooks == ["evaluate"]
    assert cfg.buffer.capacity >= 20000
    assert cfg.buffer.batch_size >= 32
    assert cfg.buffer.sequence_length >= 16
    assert cfg.controllers.planner.horizon >= 8
    assert cfg.controllers.planner.num_samples <= 64


def test_planet_dmc_continue_budget_warm_starts_with_scheduled_eval_and_checkpoints():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_continue")

    assert result["ok"] is True
    assert result["phases"] == [
        "collect_random_seed",
        "train_world_model_seed",
        "collect_with_planner_1",
        "train_world_model_1",
        "collect_with_planner_2",
        "train_world_model_2",
        "collect_with_planner_3",
        "train_world_model_3",
        "final_eval",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_continue"],
        )

    assert cfg.experiment.name == "planet_dmc_swingup_continue"
    assert cfg.training.resume_mode == "warm_start_optimizer"
    assert cfg.training.checkpoint_frequency == 512
    assert cfg.training.eval_frequency == 2048
    assert cfg.training.num_eval_episodes == 1
    assert cfg.training.max_eval_steps == 250
    assert cfg.training.phases[0].duration_steps == 1024
    assert cfg.training.phases[1].duration_updates == 256
    assert cfg.training.phases[3].duration_updates == 256
    assert cfg.training.phases[5].duration_updates == 256
    assert cfg.training.phases[7].duration_updates == 256
    assert cfg.buffer.capacity == 20000
    assert cfg.buffer.batch_size == 32
    assert cfg.buffer.sequence_length == 25
    assert cfg.controllers.planner.horizon == 8
    assert cfg.controllers.planner.num_samples == 48
    assert cfg.controllers.planner.iterations == 2


def test_planet_dmc_100ep_budget_uses_stronger_vectorized_planner_and_data_cap():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_100ep")

    assert result["ok"] is True
    assert result["phases"] == [
        "collect_random_seed",
        "train_world_model_seed",
        "collect_with_planner_1",
        "train_world_model_1",
        "collect_with_planner_2",
        "train_world_model_2",
        "collect_with_planner_3",
        "train_world_model_3",
        "collect_with_planner_4",
        "train_world_model_4",
        "collect_with_planner_5",
        "train_world_model_5",
        "final_eval",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_100ep"],
        )

    collect_steps = sum(
        int(phase.duration_steps)
        for phase in cfg.training.phases
        if "collect" in phase.workflow_hooks
    )
    update_steps = sum(
        int(phase.duration_updates)
        for phase in cfg.training.phases
        if "update_world_model" in phase.workflow_hooks
    )

    assert cfg.experiment.name == "planet_dmc_swingup_100ep"
    assert cfg.training.resume_mode == "warm_start_optimizer"
    assert cfg.training.eval_frequency == 4096
    assert cfg.training.num_eval_episodes == 1
    assert cfg.training.max_eval_steps == 250
    assert collect_steps == 12512
    assert collect_steps / cfg.environment.max_episode_steps == 100.096
    assert update_steps == 3584
    assert cfg.buffer.capacity == 40000
    assert cfg.buffer.batch_size == 32
    assert cfg.buffer.sequence_length == 25
    assert cfg.controllers.planner.horizon == 12
    assert cfg.controllers.planner.num_samples == 256
    assert cfg.controllers.planner.top_k == 32
    assert cfg.controllers.planner.iterations == 4


def test_planet_dmc_to_2000ep_budget_collects_remaining_paper_scale_data():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_to_2000ep")

    assert result["ok"] is True

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_to_2000ep"],
        )

    collect_steps = sum(
        int(phase.duration_steps)
        for phase in cfg.training.phases
        if "collect" in phase.workflow_hooks
    )
    update_steps = sum(
        int(phase.duration_updates)
        for phase in cfg.training.phases
        if "update_world_model" in phase.workflow_hooks
    )

    assert cfg.experiment.name == "planet_dmc_swingup_to_2000ep"
    assert cfg.training.resume_mode == "warm_start_optimizer"
    assert cfg.training.checkpoint_frequency == 16384
    assert cfg.training.eval_frequency == 32768
    assert cfg.training.num_eval_episodes == 1
    assert collect_steps == 234160
    assert collect_steps / cfg.environment.max_episode_steps == 1873.28
    assert update_steps == 66560
    assert cfg.buffer.capacity == 260000
    assert cfg.controllers.planner.horizon == 12
    assert cfg.controllers.planner.num_samples == 256
    assert cfg.controllers.planner.top_k == 32
    assert cfg.controllers.planner.iterations == 4


def test_planet_dmc_to_2000ep_vectorized_budget_preserves_data_budget_with_parallel_envs():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_to_2000ep_vec")

    assert result["ok"] is True
    assert result["targets"]["environment"] == "src.environments.dmc_vectorized_wrapper.DMCVectorizedWrapper"

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_to_2000ep_vec"],
        )

    collect_steps = sum(
        int(phase.duration_steps)
        for phase in cfg.training.phases
        if "collect" in phase.workflow_hooks
    )

    assert cfg.experiment.name == "planet_dmc_swingup_to_2000ep_vec"
    assert cfg.environment.num_envs == 4
    assert cfg.environment.max_episode_steps == 125
    assert cfg.training.num_eval_episodes == 4
    assert collect_steps == 234160
    assert collect_steps / cfg.environment.max_episode_steps == 1873.28
    assert cfg.buffer.capacity == 260000
    assert cfg.controllers.planner.num_samples == 256


def test_planet_dmc_paper_authentic_budget_matches_core_planet_values():
    result = validate_experiment_config("planet_dmc_cartpole_swingup", budget="planet_dmc_paper_authentic")

    assert result["ok"] is True
    assert result["targets"]["environment"] == "src.environments.dmc_vectorized_wrapper.DMCVectorizedWrapper"

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=planet_dmc_cartpole_swingup", "budget=planet_dmc_paper_authentic"],
        )

    collect_steps = sum(
        int(phase.duration_steps)
        for phase in cfg.training.phases
        if "collect" in phase.workflow_hooks
    )
    update_steps = sum(
        int(phase.duration_updates)
        for phase in cfg.training.phases
        if "update_world_model" in phase.workflow_hooks
    )

    assert cfg.experiment.name == "planet_dmc_swingup_paper_authentic"
    assert cfg.environment.num_envs == 20
    assert cfg.training.num_eval_episodes == 20
    assert cfg._dims.deterministic == 200
    assert cfg._dims.stochastic == 30
    assert cfg._dims.representation == 230
    assert cfg._dims.head_hidden == 300
    assert cfg.algorithm.reward_loss_scale == 10.0
    assert cfg.algorithm.free_nats == 3.0
    assert cfg.algorithm.max_grad_norm == 1000.0
    assert cfg.buffer.batch_size == 50
    assert cfg.buffer.sequence_length == 50
    assert cfg.controllers.planner.horizon == 12
    assert cfg.controllers.planner.num_samples == 1000
    assert cfg.controllers.planner.top_k == 100
    assert cfg.controllers.planner.iterations == 10
    assert cfg.components.reward_predictor.hidden_dim == 300
    assert cfg.components.reward_predictor.num_layers == 3
    assert collect_steps == 128000
    assert collect_steps / cfg.environment.max_episode_steps == 1024
    assert update_steps == 50000
