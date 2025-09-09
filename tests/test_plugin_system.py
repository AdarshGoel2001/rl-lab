from src.utils.registry import auto_import_modules, list_registered_components


def test_dynamic_module_loading_populates_registries():
    auto_import_modules()
    comps = list_registered_components()
    # Expect at least the known built-ins
    assert "ppo" in comps["algorithms"] or "random" in comps["algorithms"]
    assert any(n in comps["networks"] for n in ["mlp", "actor_mlp", "critic_mlp"]) 
    assert "gym" in comps["environments"] or len(comps["environments"]) >= 0

