import imprint


def test_version() -> None:
    assert isinstance(imprint.__version__, str)
