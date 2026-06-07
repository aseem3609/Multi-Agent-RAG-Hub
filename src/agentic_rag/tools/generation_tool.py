"""Answer-generation helpers for each route (vectorstore / websearch / generate).

All functions return plain answer text. Citations are assembled by the crew
orchestrator from the structured retrieval/search results.
"""

from __future__ import annotations

from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.llm import complete
from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)

_IDK = "I don't know based on the available information."

_GROUNDED_SYSTEM = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "provided context. Cite sources using their bracket numbers like [1], [2]. "
    "If the context does not contain the answer, reply exactly: "
    f"'{_IDK}' Do not invent facts."
)

_GENERATE_SYSTEM = (
    "You are a knowledgeable assistant. Answer the user's question clearly and "
    "concisely from your own knowledge. If you are not confident, say so."
)


def _history_block(history: list[dict[str, str]] | None) -> str:
    """Render short chat history into a compact prompt preamble."""
    if not history:
        return ""
    lines = [f"{turn['role']}: {turn['content']}" for turn in history[-6:]]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def generate_grounded_answer(
    question: str,
    context: str,
    *,
    history: list[dict[str, str]] | None = None,
    config: AppConfig | None = None,
) -> str:
    """Answer strictly from supplied context (used by vectorstore/websearch).

    Args:
        question: The user question.
        context: The retrieval/search context block.
        history: Optional short chat history for follow-ups.
        config: Optional config override.

    Returns:
        The grounded answer text, or an honest "I don't know".
    """
    config = config or get_config()
    if not context.strip():
        return _IDK

    user = f"{_history_block(history)}Context:\n{context}\n\nQuestion: {question}"
    return complete(
        [
            {"role": "system", "content": _GROUNDED_SYSTEM},
            {"role": "user", "content": user},
        ],
        config=config,
    ).strip()


def generate_answer(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    config: AppConfig | None = None,
) -> str:
    """Answer from the LLM's own knowledge (the ``generate`` route).

    Args:
        question: The user question.
        history: Optional short chat history.
        config: Optional config override.

    Returns:
        The generated answer text.
    """
    config = config or get_config()
    user = f"{_history_block(history)}Question: {question}"
    return complete(
        [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user", "content": user},
        ],
        config=config,
    ).strip()
