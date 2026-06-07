import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from scripts.research.export_tensorboard_run import export_tensorboard_run


def test_export_tensorboard_run_writes_scalar_summary_csv_and_plots(tmp_path):
    run_dir = tmp_path / "experiment"
    writer = SummaryWriter(log_dir=str(run_dir / "runs" / "synthetic"))
    writer.add_scalar("train/world_model/total_loss", 3.0, 1)
    writer.add_scalar("train/world_model/total_loss", 1.5, 2)
    writer.add_scalar("train/world_model/kl_loss", 0.2, 2)
    writer.add_scalar("collect/mean_step_reward", 0.4, 1)
    writer.add_scalar("eval/return_mean", 12.0, 2)
    writer.close()

    out_dir = tmp_path / "analysis"

    summary = export_tensorboard_run(run_dir, out_dir=out_dir)

    assert summary["tags"]["train/world_model/total_loss"]["count"] == 2
    assert summary["tags"]["train/world_model/total_loss"]["last_value"] == 1.5
    assert summary["tags"]["eval/return_mean"]["max"] == 12.0
    assert (out_dir / "scalar_summary.json").exists()
    assert (out_dir / "scalars.csv").exists()
    assert (out_dir / "loss_curves.png").exists()
    assert (out_dir / "collect_curves.png").exists()
    assert (out_dir / "eval_curves.png").exists()
    assert (out_dir / "eval_return_curves.png").exists()

    persisted = json.loads((out_dir / "scalar_summary.json").read_text())
    assert "train/world_model/kl_loss" in persisted["tags"]
