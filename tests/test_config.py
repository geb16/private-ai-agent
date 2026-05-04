"""Tests for runtime configuration parsing and validation."""

from __future__ import annotations

import pytest

from private_agent.config import load_settings


@pytest.fixture(autouse=True)
def clear_private_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests run with a clean PRIVATE_AGENT_* environment."""
    keys_to_clear = [
        "PRIVATE_AGENT_LLM_PROVIDER",
        "PRIVATE_AGENT_EMBEDDING_PROVIDER",
        "PRIVATE_AGENT_CHUNK_SIZE",
        "PRIVATE_AGENT_CHUNK_OVERLAP",
        "PRIVATE_AGENT_RETRIEVER_K",
    ]
    for key in keys_to_clear:
        monkeypatch.delenv(key, raising=False)


def test_default_settings_are_valid() -> None:
    """Default settings should build successfully."""
    settings = load_settings()
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.retriever_k >= 1


def test_invalid_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown provider should raise a clear validation error."""
    monkeypatch.setenv("PRIVATE_AGENT_LLM_PROVIDER", "unknown_provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        load_settings()


def test_chunk_overlap_must_be_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk overlap equal to chunk size should fail validation."""
    monkeypatch.setenv("PRIVATE_AGENT_CHUNK_SIZE", "500")
    monkeypatch.setenv("PRIVATE_AGENT_CHUNK_OVERLAP", "500")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        load_settings()
