from pathlib import Path

from src.utils.logger import create_logger


def test_logger_handles_missing_optional_backends(tmp_path: Path):
    # Even if tensorboard/wandb not installed, should not crash
    logger = create_logger(
        experiment_dir=tmp_path,
        logging_config={
            "terminal": False,
            "tensorboard": True,
            "wandb_enabled": True,
            "wandb": {"project": "dummy"},
        },
        experiment_config={"experiment": {"name": "x"}},
    )
    logger.log_metrics({"a": 1.0}, 0)
    logger.finish()
    assert True

