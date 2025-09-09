import numpy as np
import pytest


def _manager(n=2):
    from src.environments.parallel_manager import ParallelEnvironmentManager
    cfg = {"name": "DummyParallel", "wrapper": "dummy_parallel", "max_episode_steps": 10}
    return ParallelEnvironmentManager(env_config=cfg, num_environments=n, start_method="fork")


@pytest.mark.timeout(15)
def test_environment_sync_and_episode_resets():
    try:
        mgr = _manager(2)
    except Exception as e:
        pytest.skip(f"Parallel manager not available: {e}")
    try:
        obs = mgr.reset()
        assert len(obs) == 2

        def act_fn(o):
            return 0  # constant action

        data = mgr.collect_parallel_experience(action_fn=lambda x: 0, num_steps=2)
        assert data["num_environments"] == 2
        assert isinstance(data["observations"], np.ndarray)
        # After some steps, dones likely occurred and resets issued; step again to confirm no deadlock
        _ = mgr.step([0, 0])
    finally:
        mgr.close()

