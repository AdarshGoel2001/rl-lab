from scripts.validate_experiment import validate_experiment_config


def test_validate_experiment_config_reports_resolved_targets():
    result = validate_experiment_config("planet_cartpole", budget="planet_tiny")

    assert result["ok"] is True
    assert result["experiment"] == "planet_cartpole"
    assert result["budget"] == "planet_tiny"
    assert result["targets"]["workflow"] == "src.workflows.planet.PlaNetWorkflow"
    assert (
        result["targets"]["components.representation_learner"]
        == "src.components.representation_learners.rssm.RSSMRepresentationLearner"
    )
    assert result["targets"]["controllers.actor"] == "src.components.controllers.random_policy.RandomPolicyController"
    assert result["targets"]["controllers.planner"] == "src.components.controllers.mpc_planner.MPCPlanner"
    assert result["phases"] == ["data_collection", "train_world_model", "planning_eval"]
    assert result["errors"] == []


def test_validate_experiment_config_accepts_planner_override():
    result = validate_experiment_config(
        "planet_cartpole",
        budget="planet_tiny",
        extra_overrides=["controllers.planner.horizon=5"],
    )

    assert result["ok"] is True
    assert result["targets"]["controllers.planner"] == "src.components.controllers.mpc_planner.MPCPlanner"
    assert result["errors"] == []


def test_validate_experiment_config_reports_dreamer_targets_and_phases():
    result = validate_experiment_config("dreamer_dmc_cartpole_swingup", budget="dreamer_tiny")

    assert result["ok"] is True
    assert result["experiment"] == "dreamer_dmc_cartpole_swingup"
    assert result["budget"] == "dreamer_tiny"
    assert result["targets"]["workflow"] == "src.workflows.dreamer.DreamerV1Workflow"
    assert result["targets"]["controllers.actor"] == "src.components.controllers.dreamer_actor.DreamerActor"
    assert result["targets"]["controllers.critic"] == "src.components.controllers.dreamer_critic.DreamerCritic"
    assert result["phases"] == [
        "collect_random_seed",
        "train_world_model_seed",
        "train_controller_seed",
        "collect_with_actor_1",
        "train_world_model_1",
        "train_controller_1",
        "final_eval",
    ]
    assert result["errors"] == []
