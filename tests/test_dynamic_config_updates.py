from src.utils.config import ConfigManager


def test_config_manager_update(tmp_path):
    import yaml
    cfg = {
        "experiment": {"name": "e", "seed": 1},
        "algorithm": {"name": "ppo", "lr": 1e-3},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(cfg))

    cm = ConfigManager(p)
    cm.update_config({"algorithm": {"lr": 2e-3}, "experiment": {"seed": 42}})
    assert cm.config.algorithm.lr == 2e-3
    assert cm.config.experiment.seed == 42

