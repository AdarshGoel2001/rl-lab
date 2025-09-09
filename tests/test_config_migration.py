import yaml

from src.utils.config import load_config


def test_unknown_fields_pass_through_for_flexible_sections(tmp_path):
    cfg = {
        "experiment": {"name": "exp"},
        "algorithm": {"name": "ppo", "new_field": 123},
        "network": {"type": "mlp", "input_dim": 4, "output_dim": 2, "legacy_option": True},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    c = load_config(p)
    assert getattr(c.algorithm, "new_field", None) == 123
    assert getattr(c.network, "legacy_option", None) is True

