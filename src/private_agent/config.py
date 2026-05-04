"""Runtime configuration helpers for the private enterprise agent.

This module centralizes environment-driven settings so development and production
use the same code path with different environment values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv

LLMProvider = Literal["ollama", "openai"]
EmbeddingProvider = Literal["ollama", "openai"]

_SUPPORTED_LLM_PROVIDERS = {"ollama", "openai"}
_SUPPORTED_EMBEDDING_PROVIDERS = {"ollama", "openai"}

_ENV_PREFIX = "PRIVATE_AGENT_"


def _env_key(name: str) -> str:
    """Return a fully-qualified environment variable name."""
    return f"{_ENV_PREFIX}{name}"


def _read_env_str(name: str, default: str) -> str:
    """Read a trimmed string from environment, falling back to default."""
    raw_value = os.getenv(_env_key(name))
    if raw_value is None:
        return default

    cleaned = raw_value.strip()
    return cleaned if cleaned else default


def _read_env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer from environment with validation."""
    raw_value = os.getenv(_env_key(name))
    if raw_value is None:
        return default

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_env_key(name)} must be an integer.") from exc

    if parsed_value < minimum:
        raise ValueError(f"{_env_key(name)} must be >= {minimum}.")

    return parsed_value


def _read_env_float(name: str, default: float, minimum: float = 0.0) -> float:
    """Read a float from environment with validation."""
    raw_value = os.getenv(_env_key(name))
    if raw_value is None:
        return default

    try:
        parsed_value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_env_key(name)} must be a number.") from exc

    if parsed_value < minimum:
        raise ValueError(f"{_env_key(name)} must be >= {minimum}.")

    return parsed_value


@dataclass(frozen=True, slots=True)
class Settings:
    """Strongly typed runtime settings for ingestion and question answering."""

    base_dir: Path
    data_dir: Path
    vector_db_dir: Path
    collection_name: str

    llm_provider: LLMProvider
    llm_model: str
    llm_temperature: float
    llm_timeout_seconds: int

    embedding_provider: EmbeddingProvider
    embedding_model: str

    chunk_size: int
    chunk_overlap: int
    retriever_k: int

    log_level: str


def load_settings() -> Settings:
    """Create validated settings from environment variables.

    Expected environment variables:
    - PRIVATE_AGENT_LLM_PROVIDER
    - PRIVATE_AGENT_LLM_MODEL
    - PRIVATE_AGENT_EMBEDDING_PROVIDER
    - PRIVATE_AGENT_EMBEDDING_MODEL
    - PRIVATE_AGENT_DATA_DIR
    - PRIVATE_AGENT_VECTOR_DB_DIR
    - PRIVATE_AGENT_CHUNK_SIZE
    - PRIVATE_AGENT_CHUNK_OVERLAP
    - PRIVATE_AGENT_RETRIEVER_K
    - PRIVATE_AGENT_LOG_LEVEL
    """
    # Load local `.env` for development without affecting container/prod env behavior.
    load_dotenv()

    base_dir = Path(__file__).resolve().parents[2] # levels up to the project root

    data_dir = Path(_read_env_str("DATA_DIR", str(base_dir / "data")))
    vector_db_dir = Path(_read_env_str("VECTOR_DB_DIR", str(base_dir / "chroma_db")))

    llm_provider_raw = _read_env_str("LLM_PROVIDER", "ollama").lower()
    embedding_provider_raw = _read_env_str("EMBEDDING_PROVIDER", "ollama").lower()

    if llm_provider_raw not in _SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{llm_provider_raw}'. "
            f"Supported values: {sorted(_SUPPORTED_LLM_PROVIDERS)}"
        )

    if embedding_provider_raw not in _SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported embedding provider '{embedding_provider_raw}'. "
            f"Supported values: {sorted(_SUPPORTED_EMBEDDING_PROVIDERS)}"
        )

    chunk_size = _read_env_int("CHUNK_SIZE", 700, minimum=100)
    chunk_overlap = _read_env_int("CHUNK_OVERLAP", 120, minimum=0)

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"{_env_key('CHUNK_OVERLAP')} must be smaller than {_env_key('CHUNK_SIZE')}."
        )

    return Settings(
        base_dir=base_dir,
        data_dir=data_dir,
        vector_db_dir=vector_db_dir,
        collection_name=_read_env_str("COLLECTION_NAME", "enterprise_knowledge"),
        llm_provider=cast(LLMProvider, llm_provider_raw),
        llm_model=_read_env_str("LLM_MODEL", "mistral-small3.2"),
        llm_temperature=_read_env_float("LLM_TEMPERATURE", 0.0, minimum=0.0),
        llm_timeout_seconds=_read_env_int("LLM_TIMEOUT_SECONDS", 120, minimum=5),
        embedding_provider=cast(EmbeddingProvider, embedding_provider_raw),
        embedding_model=_read_env_str("EMBEDDING_MODEL", "nomic-embed-text"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_k=_read_env_int("RETRIEVER_K", 4, minimum=1),
        log_level=_read_env_str("LOG_LEVEL", "INFO").upper(),
    )
