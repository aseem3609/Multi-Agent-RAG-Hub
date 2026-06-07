"""Typed application configuration loaded from environment variables.

All tunable values (models, API keys, paths, ``top_k`` …) live here so that
nothing is hardcoded elsewhere in the codebase. Configuration is loaded once
and cached via :func:`get_config`.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM/embedding providers routed through LiteLLM."""

    OPENAI = "openai"
    AZURE = "azure"
    GROQ = "groq"


class AppConfig(BaseSettings):
    """Strongly-typed application configuration.

    Values are read from environment variables (and an optional ``.env`` file).
    Sensible defaults are provided so the system runs end-to-end with only an
    ``OPENAI_API_KEY`` set.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider selection ------------------------------------------------
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE", ge=0.0, le=2.0)

    # --- Credentials -------------------------------------------------------
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    azure_api_key: str | None = Field(default=None, alias="AZURE_API_KEY")
    azure_api_base: str | None = Field(default=None, alias="AZURE_API_BASE")
    azure_api_version: str | None = Field(default=None, alias="AZURE_API_VERSION")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    # --- Vector store / ingestion -----------------------------------------
    chroma_persist_dir: Path = Field(default=Path("./.chroma"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="agentic_rag", alias="CHROMA_COLLECTION")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=4, alias="TOP_K", gt=0)
    grade_threshold: float = Field(default=0.5, alias="GRADE_THRESHOLD", ge=0.0, le=1.0)

    # --- Application -------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, value: int, info) -> int:
        """Ensure overlap never exceeds (or equals) the chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def web_search_enabled(self) -> bool:
        """Whether a usable Tavily API key is configured."""
        return bool(self.tavily_api_key)

    def litellm_model(self) -> str:
        """Return the LiteLLM-compatible model identifier for the LLM."""
        return self.llm_model

    def litellm_embedding_model(self) -> str:
        """Return the LiteLLM-compatible model identifier for embeddings."""
        return self.embedding_model


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the cached, singleton application configuration.

    Returns:
        The validated :class:`AppConfig` instance.
    """
    return AppConfig()
