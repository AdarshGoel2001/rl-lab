from pathlib import Path

from src.core.trainer import create_trainer_from_config


def test_experiment_dirs_created(tmp_path):
    import yaml
    cfg = {
        "experiment": {"name": "lifecycle", "seed": 0, "device": "cpu"},
        "algorithm": {"name": "random"},
        "environment": {"name": "Dummy", "wrapper": "dummy_single"},
        "network": {"type": "mlp", "input_dim": 4, "output_dim": 2},
        "buffer": {"type": "trajectory", "capacity": 32, "batch_size": 8},
        "training": {"total_timesteps": 8, "eval_frequency": 8, "checkpoint_frequency": 1000},
        "logging": {"terminal": False, "tensorboard": False, "wandb_enabled": False},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    exp_dir = tmp_path / "exp"
    trainer = create_trainer_from_config(str(p), experiment_dir=str(exp_dir))
    assert (exp_dir / "logs").exists()
    assert (exp_dir / "checkpoints").exists()
    trainer.cleanup()

