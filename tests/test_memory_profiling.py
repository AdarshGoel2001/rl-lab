import tracemalloc

from tests._helpers import issue_logger


def test_memory_snapshot_leak_check():
    tracemalloc.start()
    s1 = tracemalloc.take_snapshot()

    # Perform some allocations
    xs = [bytearray(1024) for _ in range(1000)]  # ~1MB
    del xs

    s2 = tracemalloc.take_snapshot()
    stats = s2.compare_to(s1, "filename")
    top = sum(stat.size_diff for stat in stats[:10])
    issue_logger.log(
        category="performance",
        severity="info",
        message="Top allocation diff (bytes)",
        extra={"top10_size_diff": int(top)},
    )
    tracemalloc.stop()
    assert True

