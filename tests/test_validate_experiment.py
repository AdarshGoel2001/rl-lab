from scripts.validate_experiment import validate_experiment_config


def test_validate_experiment_config_reports_resolved_targets():
    result = validate_experiment_config("og_wm_carracing", budget="tiny")

    assert result["ok"] is True
    assert result["experiment"] == "og_wm_carracing"
    assert result["budget"] == "tiny"
    assert result["targets"]["workflow"] == "src.workflows.og_wm.OriginalWorldModelsWorkflow"
    assert (
        result["targets"]["components.vae"]
        == "src.components.representation_learners.conv_vae.ConvVAERepresentationLearner"
    )
    assert result["targets"]["components.dynamics_model"] == "src.components.dynamics.mdn_rnn.MDNRNNDynamics"
    assert result["targets"]["controllers.actor"] == "src.components.controllers.random_policy.RandomPolicyController"
    assert result["phases"] == ["data_collection", "converge_vae", "converge_mdn"]
    assert result["errors"] == []


def test_validate_experiment_config_accepts_dynamics_override():
    result = validate_experiment_config(
        "og_wm_carracing",
        budget="tiny",
        extra_overrides=["components/dynamics_model=gaussian_gru"],
    )

    assert result["ok"] is True
    assert (
        result["targets"]["components.dynamics_model"]
        == "src.components.dynamics.gaussian_gru.GaussianGRUDynamics"
    )
    assert result["errors"] == []
