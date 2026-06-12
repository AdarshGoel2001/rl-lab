from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.validate_experiment import validate_experiment_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _compose_dreamer_tiny():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=["+experiment=dreamer_dmc_cartpole_swingup", "budget=dreamer_tiny"],
        )


def _compose_dreamer_100ep():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=["+experiment=dreamer_dmc_cartpole_swingup", "budget=dreamer_dmc_100ep"],
        )


def test_dreamer_dmc_cartpole_config_resolves_core_targets():
    result = validate_experiment_config("dreamer_dmc_cartpole_swingup", budget="dreamer_tiny")

    assert result["ok"] is True
    assert result["targets"]["workflow"] == "src.workflows.dreamer.DreamerV1Workflow"
    assert (
        result["targets"]["components.representation_learner"]
        == "src.components.representation_learners.rssm.RSSMRepresentationLearner"
    )
    assert result["targets"]["components.reward_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["targets"]["components.continue_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["targets"]["components.observation_predictor"] == "src.components.prediction_heads.mlp.MLPHead"
    assert result["targets"]["controllers.seed_actor"] == "src.components.controllers.random_policy.RandomPolicyController"
    assert result["targets"]["controllers.actor"] == "src.components.controllers.dreamer_actor.DreamerActor"
    assert result["targets"]["controllers.critic"] == "src.components.controllers.dreamer_critic.DreamerCritic"
    assert result["targets"]["environment"] == "src.environments.dmc_wrapper.DMCWrapper"
    assert result["errors"] == []


def test_dreamer_dmc_cartpole_tiny_dimensions_and_optimizer_roles():
    cfg = _compose_dreamer_tiny()

    assert cfg.environment.name == "cartpole_swingup"
    assert cfg.environment.from_pixels is False
    assert cfg._dims.observation == 5
    assert cfg._dims.encoder_output == 5
    assert cfg._dims.action == 1
    assert cfg._dims.representation == cfg._dims.deterministic + cfg._dims.stochastic

    assert cfg.components.representation_learner.feature_dim == cfg._dims.encoder_output
    assert cfg.components.representation_learner.action_dim == cfg._dims.action
    assert cfg.components.observation_predictor.input_dim == cfg._dims.representation
    assert cfg.components.observation_predictor.output_dim == cfg._dims.observation
    assert cfg.components.reward_predictor.input_dim == cfg._dims.representation
    assert cfg.components.continue_predictor.input_dim == cfg._dims.representation

    assert set(cfg.optimizers.keys()) == {"world_model", "actor", "critic"}
    assert cfg.optimizers.world_model.lr == cfg.algorithm.world_model_lr
    assert cfg.optimizers.actor.lr == cfg.algorithm.actor_lr
    assert cfg.optimizers.critic.lr == cfg.algorithm.critic_lr


def test_dreamer_dmc_cartpole_controller_roles_and_action_bounds():
    cfg = _compose_dreamer_tiny()

    assert cfg.controllers.seed_actor.action_dim == 1
    assert cfg.controllers.seed_actor.action_low == [-1.0]
    assert cfg.controllers.seed_actor.action_high == [1.0]
    assert cfg.controllers.actor.action_dim == 1
    assert cfg.controllers.actor.latent_dim == cfg._dims.representation
    assert cfg.controllers.actor.action_low == [-1.0]
    assert cfg.controllers.actor.action_high == [1.0]
    assert cfg.controllers.critic.latent_dim == cfg._dims.representation


def test_dreamer_dmc_cartpole_phase_hooks_cover_world_model_controller_and_eval():
    cfg = _compose_dreamer_tiny()

    phase_names = [phase.name for phase in cfg.training.phases]
    hooks = [hook for phase in cfg.training.phases for hook in phase.workflow_hooks]

    assert phase_names == [
        "collect_random_seed",
        "train_world_model_seed",
        "train_controller_seed",
        "collect_with_actor_1",
        "train_world_model_1",
        "train_controller_1",
        "final_eval",
    ]
    assert {"collect", "update_world_model", "update_controller", "evaluate"}.issubset(set(hooks))
    assert cfg.training.phases[0].controller == "seed_actor"
    assert cfg.training.phases[3].controller == "actor"
    assert cfg.training.phases[-1].workflow_hooks == ["evaluate"]


def test_dreamer_dmc_100ep_uses_larger_state_model_and_five_eval_episodes():
    cfg = _compose_dreamer_100ep()

    assert cfg._dims.stochastic == 30
    assert cfg._dims.deterministic == 200
    assert cfg._dims.representation == 230
    assert cfg.controllers.actor.hidden_dim == 300
    assert cfg.controllers.actor.num_layers == 3
    assert cfg.controllers.critic.hidden_dim == 300
    assert cfg.controllers.critic.num_layers == 3
    assert cfg.algorithm.imagination_horizon == 15
    assert cfg.buffer.capacity == 50000
    assert cfg.buffer.batch_size == 50
    assert cfg.buffer.sequence_length == 50
    assert cfg.training.num_eval_episodes == 5
    assert cfg.training.resume_mode == "warm_start_optimizer"
