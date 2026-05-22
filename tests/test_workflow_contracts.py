from src.workflows.utils.base import CollectResult


def test_collect_result_accepts_optional_extras():
    result = CollectResult(steps=3, extras={"replay": {"trajectory": {"observations": []}}})

    assert result.steps == 3
    assert result.extras["replay"]["trajectory"]["observations"] == []
