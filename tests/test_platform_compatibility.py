from src.utils.config import resolve_device


def test_auto_device_resolution():
    dev = resolve_device("auto")
    assert dev in {"cpu", "cuda", "mps"}

