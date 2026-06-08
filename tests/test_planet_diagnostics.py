import csv
import json

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing import event_accumulator

from scripts.research.diagnostics.diagnose_planet_checkpoint import (
    compute_horizon_return_metrics,
    compute_regression_metrics,
    diagnostic_orchestrator_dir,
    validate_planet_checkpoint_schema,
    write_diagnostic_outputs,
)


def test_compute_regression_metrics_reports_error_bias_and_correlation():
    predicted = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    actual = np.asarray([0.0, 2.0, 2.0, 2.0], dtype=np.float32)

    metrics = compute_regression_metrics(predicted, actual, prefix="reward")

    assert metrics["reward/count"] == 4
    assert metrics["reward/mse"] == 1.25
    assert metrics["reward/mae"] == 0.75
    assert metrics["reward/bias"] == 0.25
    assert metrics["reward/corr"] > 0.25


def test_compute_regression_metrics_uses_nan_correlation_for_constant_arrays():
    metrics = compute_regression_metrics(
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        prefix="reward",
    )

    assert np.isnan(metrics["reward/corr"])


def test_compute_horizon_return_metrics_uses_cumulative_rewards():
    predicted = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    actual = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    rows, summary = compute_horizon_return_metrics(predicted, actual, horizons=[1, 2, 3])

    assert [row["horizon"] for row in rows] == [1, 2, 3]
    assert rows[0]["return_mse"] == 0.0
    assert rows[1]["return_mae"] == 1.0
    assert summary["open_loop/h3_return_bias"] == -2.5


def test_write_diagnostic_outputs_writes_json_csv_and_plots(tmp_path):
    summary = {
        "checkpoint": "checkpoint.pt",
        "reward/mse": 0.25,
        "open_loop/h2_return_mae": 1.5,
    }
    reward_rows = [
        {"step": 0, "env": 0, "predicted_reward": 0.25, "actual_reward": 0.5},
        {"step": 1, "env": 0, "predicted_reward": 0.75, "actual_reward": 1.0},
    ]
    horizon_rows = [
        {"horizon": 1, "return_mse": 0.25, "return_mae": 0.5, "return_bias": -0.5, "return_corr": 1.0},
        {"horizon": 2, "return_mse": 2.25, "return_mae": 1.5, "return_bias": -1.5, "return_corr": 1.0},
    ]

    outputs = write_diagnostic_outputs(
        out_dir=tmp_path,
        summary=summary,
        reward_rows=reward_rows,
        horizon_rows=horizon_rows,
    )

    assert (tmp_path / "diagnostics_summary.json").exists()
    assert (tmp_path / "reward_calibration.csv").exists()
    assert (tmp_path / "open_loop_horizon_metrics.csv").exists()
    assert (tmp_path / "reward_pred_vs_actual.png").exists()
    assert (tmp_path / "open_loop_errors.png").exists()
    assert json.loads((tmp_path / "diagnostics_summary.json").read_text())["reward/mse"] == 0.25
    with (tmp_path / "reward_calibration.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert outputs["summary"].name == "diagnostics_summary.json"


def test_write_diagnostic_outputs_logs_scalars_to_tensorboard(tmp_path):
    summary = {
        "checkpoint": "checkpoint.pt",
        "checkpoint_global_step": 123,
        "reward/mse": 0.25,
        "reward/corr": 0.9,
        "open_loop/h2_return_mae": 1.5,
        "policy": "random",
    }
    reward_rows = [
        {"step": 0, "env": 0, "predicted_reward": 0.25, "actual_reward": 0.5},
        {"step": 1, "env": 0, "predicted_reward": 0.75, "actual_reward": 1.0},
    ]
    horizon_rows = [
        {"horizon": 1, "return_mse": 0.25, "return_mae": 0.5, "return_bias": -0.5, "return_corr": 1.0},
        {"horizon": 2, "return_mse": 2.25, "return_mae": 1.5, "return_bias": -1.5, "return_corr": 1.0},
    ]
    tb_dir = tmp_path / "runs" / "diagnostics"

    outputs = write_diagnostic_outputs(
        out_dir=tmp_path / "analysis",
        summary=summary,
        reward_rows=reward_rows,
        horizon_rows=horizon_rows,
        tensorboard_logdir=tb_dir,
    )

    event_files = list(tb_dir.glob("events.out.tfevents*"))
    assert event_files
    accumulator = event_accumulator.EventAccumulator(str(tb_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert "diagnostics/reward/mse" in scalar_tags
    assert "diagnostics/reward/corr" in scalar_tags
    assert "diagnostics/open_loop/return_mae" in scalar_tags
    assert "diagnostics/open_loop/h2_return_mae" not in scalar_tags
    reward_event = accumulator.Scalars("diagnostics/reward/mse")[0]
    horizon_event = accumulator.Scalars("diagnostics/open_loop/return_mae")[-1]
    assert reward_event.step == 123
    assert horizon_event.step == 2
    assert horizon_event.value == 1.5
    assert outputs["tensorboard_logdir"] == tb_dir


def test_validate_planet_checkpoint_schema_rejects_missing_component_weights(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    torch.save({"global_step": 10, "components": {"reward_predictor": {}}}, checkpoint)

    with pytest.raises(ValueError, match="missing required component weights"):
        validate_planet_checkpoint_schema(checkpoint)


def test_validate_planet_checkpoint_schema_accepts_required_component_weights(tmp_path):
    checkpoint = tmp_path / "good.pt"
    torch.save(
        {
            "global_step": 10,
            "components": {
                "representation_learner": {"weight": torch.zeros(1)},
                "reward_predictor": {"weight": torch.zeros(1)},
            },
        },
        checkpoint,
    )

    metadata = validate_planet_checkpoint_schema(checkpoint)

    assert metadata["global_step"] == 10
    assert metadata["component_names"] == ["representation_learner", "reward_predictor"]


def test_diagnostic_orchestrator_dir_lives_under_diagnostic_output(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    out_dir = tmp_path / "run" / "diagnostics" / "planet_reward_open_loop"

    assert diagnostic_orchestrator_dir(checkpoint=checkpoint, out_dir=out_dir) == out_dir / "_orchestrator"
