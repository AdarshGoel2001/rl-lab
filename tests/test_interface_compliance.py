import inspect

from src.utils.registry import auto_import_modules, ALGORITHM_REGISTRY, NETWORK_REGISTRY, BUFFER_REGISTRY


def test_components_follow_base_interfaces():
    auto_import_modules()

    # Algorithms: must implement act and update
    for name, cls in ALGORITHM_REGISTRY.items():
        for method in ("act", "update"):
            assert hasattr(cls, method), f"Algorithm {name} missing {method}"

    # Networks: must implement forward
    for name, cls in NETWORK_REGISTRY.items():
        assert hasattr(cls, "forward"), f"Network {name} missing forward"

    # Buffers: must implement add, sample, clear
    for name, cls in BUFFER_REGISTRY.items():
        for method in ("add", "sample", "clear"):
            assert hasattr(cls, method), f"Buffer {name} missing {method}"

