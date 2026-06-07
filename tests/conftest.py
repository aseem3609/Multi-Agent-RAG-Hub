"""Shared pytest fixtures and offline environment setup."""

from __future__ import annotations

import os

import pytest

# Ensure config never depends on a developer's real environment / .env file.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "")


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset the cached config between tests so env overrides take effect."""
    from agentic_rag.core.config import get_config

    get_config.cache_clear()
    yield
    get_config.cache_clear()


class FakeStore:
    """A minimal in-memory stand-in for :class:`VectorStore`."""

    def __init__(self, chunks=None):
        self._chunks = chunks or []

    def count(self) -> int:
        return len(self._chunks)

    def query(self, question: str, *, top_k: int | None = None):
        return self._chunks[: (top_k or len(self._chunks))]


@pytest.fixture
def fake_store():
    """Factory fixture producing a :class:`FakeStore`."""
    return FakeStore
