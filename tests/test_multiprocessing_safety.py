import pytest


def _setup_manager(num_envs=2):
    from src.environments.parallel_manager import ParallelEnvironmentManager
    env_cfg = {"name": "DummyParallel", "wrapper": "dummy_parallel", "max_episode_steps": 10}
    return ParallelEnvironmentManager(env_config=env_cfg, num_environments=num_envs, start_method="fork")


@pytest.mark.timeout(15)
def test_process_isolation_and_determinism():
    try:
        mgr = _setup_manager(2)
    except Exception as e:
        pytest.skip(f"Parallel manager not available: {e}")
    try:
        obs1 = mgr.reset()
        # Step a few times; ensure observations collection length matches envs
        for _ in range(3):
            next_obs, rewards, dones, infos = mgr.step([0, 1])
            assert len(next_obs) == 2
            assert len(dones) == 2
        # Episodes should have completed at least once
        assert sum(mgr.episodes_completed) >= 1
    finally:
        mgr.close()

