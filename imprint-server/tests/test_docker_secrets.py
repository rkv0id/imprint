"""Tests for Docker secrets _FILE variable loading in cli.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_load_file_secrets_populates_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_file = tmp_path / "anthropic_key"
    secret_file.write_text("sk-ant-test-key-123")

    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(secret_file))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from imprint_server.cli import _load_file_secrets

    _load_file_secrets()

    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-key-123"


def test_load_file_secrets_strips_trailing_newline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_file = tmp_path / "key_with_newline"
    secret_file.write_text("sk-voyage-key\n")

    monkeypatch.setenv("VOYAGE_API_KEY_FILE", str(secret_file))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    from imprint_server.cli import _load_file_secrets

    _load_file_secrets()

    assert os.environ.get("VOYAGE_API_KEY") == "sk-voyage-key"


def test_load_file_secrets_noop_when_no_file_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "ANTHROPIC_API_KEY_FILE",
        "VOYAGE_API_KEY_FILE",
        "OPENAI_API_KEY_FILE",
        "IMPRINT_REDIS_URL_FILE",
    ):
        monkeypatch.delenv(var, raising=False)

    from imprint_server.cli import _load_file_secrets

    _load_file_secrets()  # must not raise


def test_load_file_secrets_raises_on_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(tmp_path / "does_not_exist"))

    from imprint_server.cli import _load_file_secrets

    with pytest.raises(SystemExit, match="could not be read"):
        _load_file_secrets()


def test_load_file_secrets_raises_on_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_file = tmp_path / "empty"
    empty_file.write_text("")

    monkeypatch.setenv("VOYAGE_API_KEY_FILE", str(empty_file))

    from imprint_server.cli import _load_file_secrets

    with pytest.raises(SystemExit, match="empty"):
        _load_file_secrets()


def test_load_file_secrets_redis_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret_file = tmp_path / "redis_url"
    secret_file.write_text("redis://localhost:6379/0")

    monkeypatch.setenv("IMPRINT_REDIS_URL_FILE", str(secret_file))
    monkeypatch.delenv("IMPRINT_REDIS_URL", raising=False)

    from imprint_server.cli import _load_file_secrets

    _load_file_secrets()

    assert os.environ.get("IMPRINT_REDIS_URL") == "redis://localhost:6379/0"
