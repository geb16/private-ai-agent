"""Factories for LLM and embedding model backends."""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from private_agent.config import Settings


def _require_openai_key() -> None:
    """Fail fast when OpenAI provider is selected without an API key."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required when PRIVATE_AGENT_*_PROVIDER is 'openai'."
        )


def create_embeddings(settings: Settings) -> Embeddings:
    """Create an embeddings client for the configured provider."""
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(model=settings.embedding_model)

    if settings.embedding_provider == "openai":
        _require_openai_key()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create a chat model client for the configured provider."""
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )

    if settings.llm_provider == "openai":
        _require_openai_key()
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=3,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
