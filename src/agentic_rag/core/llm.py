"""Provider-agnostic LLM and embedding access via LiteLLM.

This module is the single place that talks to LiteLLM. Switching providers
(OpenAI ↔ Azure ↔ Groq) only requires changing environment variables — no
code changes. It exposes:

- :func:`provider_kwargs`     : provider-specific credentials for LiteLLM calls.
- :func:`complete`            : a chat completion returning plain text.
- :func:`complete_json`       : a chat completion parsed into a Pydantic model.
- :func:`embed_texts`         : batch embeddings for a list of strings.
- :func:`build_crewai_llm`    : a configured CrewAI ``LLM`` instance.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agentic_rag.core.config import AppConfig, LLMProvider, get_config
from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM or embedding call fails irrecoverably."""


def provider_kwargs(config: AppConfig | None = None) -> dict[str, object]:
    """Return provider-specific keyword arguments for LiteLLM calls.

    Args:
        config: Optional config override (defaults to the cached config).

    Returns:
        A dict of credentials/endpoints suitable for ``litellm.completion``.

    Raises:
        LLMError: If the selected provider is missing required credentials.
    """
    config = config or get_config()
    provider = config.llm_provider

    if provider is LLMProvider.OPENAI:
        if not config.openai_api_key:
            raise LLMError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return {"api_key": config.openai_api_key}

    if provider is LLMProvider.AZURE:
        if not (config.azure_api_key and config.azure_api_base):
            raise LLMError("AZURE_API_KEY and AZURE_API_BASE are required when LLM_PROVIDER=azure")
        return {
            "api_key": config.azure_api_key,
            "api_base": config.azure_api_base,
            "api_version": config.azure_api_version,
        }

    if provider is LLMProvider.GROQ:
        if not config.groq_api_key:
            raise LLMError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return {"api_key": config.groq_api_key}

    raise LLMError(f"Unsupported provider: {provider}")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def complete(
    messages: list[dict[str, str]],
    *,
    config: AppConfig | None = None,
    temperature: float | None = None,
    response_format: dict[str, str] | None = None,
) -> str:
    """Run a chat completion and return the assistant's text content.

    Args:
        messages: OpenAI-style chat messages.
        config: Optional config override.
        temperature: Optional sampling temperature override.
        response_format: Optional LiteLLM ``response_format`` (e.g. JSON mode).

    Returns:
        The assistant message content as a string.

    Raises:
        LLMError: If the call fails after retries.
    """
    # Imported lazily so importing this module never forces a heavy dependency.
    import litellm

    config = config or get_config()
    try:
        response = litellm.completion(
            model=config.litellm_model(),
            messages=messages,
            temperature=config.llm_temperature if temperature is None else temperature,
            response_format=response_format,
            **provider_kwargs(config),
        )
        return response["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001 - normalised into LLMError below
        logger.error("LLM completion failed: %s", exc)
        raise LLMError(str(exc)) from exc


def complete_json(
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    config: AppConfig | None = None,
) -> T:
    """Run a JSON-mode completion and validate it into a Pydantic model.

    Args:
        messages: OpenAI-style chat messages. The system/user prompt must
            instruct the model to return JSON matching ``schema``.
        schema: The Pydantic model to validate the response against.
        config: Optional config override.

    Returns:
        A validated instance of ``schema``.

    Raises:
        LLMError: If the response cannot be parsed or validated.
    """
    raw = complete(
        messages,
        config=config,
        response_format={"type": "json_object"},
    )
    try:
        return schema.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse structured LLM output: %s | raw=%s", exc, raw[:500])
        raise LLMError(f"Invalid structured output: {exc}") from exc


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def embed_texts(texts: list[str], *, config: AppConfig | None = None) -> list[list[float]]:
    """Embed a batch of texts and return their vectors.

    Args:
        texts: The strings to embed.
        config: Optional config override.

    Returns:
        A list of embedding vectors aligned with ``texts``.

    Raises:
        LLMError: If the embedding call fails after retries.
    """
    import litellm

    config = config or get_config()
    if not texts:
        return []
    try:
        response = litellm.embedding(
            model=config.litellm_embedding_model(),
            input=texts,
            **provider_kwargs(config),
        )
        # LiteLLM normalises to OpenAI's response shape.
        return [item["embedding"] for item in response["data"]]
    except Exception as exc:  # noqa: BLE001 - normalised into LLMError below
        logger.error("Embedding call failed: %s", exc)
        raise LLMError(str(exc)) from exc


def build_crewai_llm(config: AppConfig | None = None):
    """Build a CrewAI ``LLM`` configured for the selected provider.

    Args:
        config: Optional config override.

    Returns:
        A ``crewai.LLM`` instance, or ``None`` if CrewAI is unavailable.
    """
    config = config or get_config()
    try:
        from crewai import LLM
    except Exception as exc:  # noqa: BLE001 - CrewAI optional at import time
        logger.warning("CrewAI LLM unavailable: %s", exc)
        return None

    return LLM(
        model=config.litellm_model(),
        temperature=config.llm_temperature,
        **provider_kwargs(config),
    )
