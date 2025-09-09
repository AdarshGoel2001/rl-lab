import yaml

from src.utils.config import load_config


def test_config_hash_stability(tmp_path):
    cfg = {"experiment": {"name": "h", "seed": 1}, "algorithm": {"name": "ppo"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(cfg))
    c1 = load_config(p)
    c2 = load_config(p)
    assert c1.get_hash() == c2.get_hash()

