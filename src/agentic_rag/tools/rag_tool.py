"""Retrieval tool: fetch and format relevant chunks from the vector store."""

from __future__ import annotations

from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.vectorstore import RetrievedChunk, VectorStore

logger = get_logger(__name__)


def retrieve_context(
    question: str,
    *,
    top_k: int | None = None,
    store: VectorStore | None = None,
    config: AppConfig | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-k most relevant chunks for a question.

    Args:
        question: The user question.
        top_k: Number of chunks (defaults to config ``top_k``).
        store: Optional vector store override (useful for tests).
        config: Optional config override.

    Returns:
        A list of :class:`RetrievedChunk`, possibly empty.
    """
    config = config or get_config()
    store = store or VectorStore(config)
    try:
        chunks = store.query(question, top_k=top_k or config.top_k)
        logger.info("Retrieved %d chunks for question", len(chunks))
        return chunks
    except Exception as exc:  # noqa: BLE001 - retrieval failure -> empty result
        logger.error("Retrieval failed: %s", exc)
        return []


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a numbered context block for prompting.

    Each block is prefixed with a bracketed index and its citation so the LLM
    can reference sources precisely.
    """
    if not chunks:
        return ""
    return "\n\n".join(
        f"[{i}] Source: {chunk.citation()}\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
