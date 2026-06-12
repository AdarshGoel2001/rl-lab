import json

from scripts.research.diagnostics.diagnose_dreamer_tensorboard import (
    load_scalar_rows,
    render_markdown_report,
    summarize_dreamer_scalars,
    write_dreamer_diagnostics,
)


def _row(tag, step, value):
    return {"tag": tag, "step": step, "wall_time": float(step), "value": value, "event_file": "events"}


def test_summarize_dreamer_scalars_reports_eval_and_controller_health():
    rows = [
        _row("eval/return_mean", 10, 100.0),
        _row("eval/return_mean", 20, 250.0),
        _row("eval/return_mean", 30, 200.0),
        _row("train/world_model/total_loss", 30, 0.5),
        _row("train/world_model/reward_loss", 30, 0.2),
        _row("controller/critic_target_mean", 30, 3.0),
        _row("controller/critic_value_mean", 30, 1.5),
        _row("controller/action_abs_mean", 30, 0.25),
        _row("controller/action_abs_max", 30, 0.7),
        _row("controller/lambda_return_mean", 30, 4.0),
        _row("controller/lambda_return_std", 30, 0.8),
        _row("controller/imagined_continue_mean", 30, 0.99),
    ]

    summary = summarize_dreamer_scalars(rows)

    assert summary["eval"]["latest_return_mean"] == 200.0
    assert summary["eval"]["best_return_mean"] == 250.0
    assert summary["eval"]["best_step"] == 20
    assert summary["world_model"]["world_model_total_loss"] == 0.5
    assert summary["controller"]["critic_target_value_gap"] == 1.5
    assert summary["controller"]["critic_target_value_abs_gap"] == 1.5
    assert "actor_action_abs_mean_high" not in summary["warnings"]


def test_summarize_dreamer_scalars_warns_on_action_saturation():
    rows = [
        _row("controller/action_abs_mean", 1, 0.95),
        _row("controller/action_abs_max", 1, 0.999),
        _row("controller/imagined_continue_mean", 1, 0.5),
    ]

    summary = summarize_dreamer_scalars(rows)

    assert "actor_action_abs_mean_high" in summary["warnings"]
    assert "actor_action_abs_max_saturated" in summary["warnings"]


def test_write_dreamer_diagnostics_writes_json_and_markdown(tmp_path):
    summary = {
        "schema_version": 1,
        "missing_tags": [],
        "warnings": ["actor_action_abs_max_saturated"],
        "eval": {"best_return_mean": 250.0, "best_step": 20, "latest_return_mean": 200.0, "latest_step": 30},
        "world_model": {"world_model_total_loss": 0.5},
        "controller": {"action_abs_max": 0.999},
    }

    outputs = write_dreamer_diagnostics(summary, tmp_path)

    assert json.loads(outputs["summary"].read_text())["eval"]["best_return_mean"] == 250.0
    report = outputs["report"].read_text()
    assert "Dreamer TensorBoard Diagnostics" in report
    assert "actor_action_abs_max_saturated" in report
    assert render_markdown_report(summary).startswith("# Dreamer")


def test_load_scalar_rows_accepts_analysis_directory(tmp_path):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    scalars_csv = analysis_dir / "scalars.csv"
    scalars_csv.write_text(
        "tag,step,wall_time,value,event_file\n"
        "eval/return_mean,10,1.5,200.0,events\n",
        encoding="utf-8",
    )

    rows = load_scalar_rows(analysis_dir)

    assert rows == [
        {
            "tag": "eval/return_mean",
            "step": 10,
            "wall_time": 1.5,
            "value": 200.0,
            "event_file": "events",
        }
    ]
