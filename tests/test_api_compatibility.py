def test_key_api_surfaces_exist():
    import src.core.trainer as tr
    import src.utils.config as cfg
    import src.environments.parallel_manager as pm

    assert hasattr(tr, "create_trainer_from_config")
    assert hasattr(cfg, "load_config") and hasattr(cfg, "apply_config_overrides")
    assert hasattr(pm, "ParallelEnvironmentManager")

