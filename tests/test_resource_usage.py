from tests._helpers import issue_logger, has_torch_cuda, has_torch_mps


def test_device_presence_logging():
    gpu = has_torch_cuda()
    mps = has_torch_mps()
    issue_logger.log(
        category="resources",
        severity="info",
        message="Device availability",
        extra={"cuda": gpu, "mps": mps},
    )
    assert True

