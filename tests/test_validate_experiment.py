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
