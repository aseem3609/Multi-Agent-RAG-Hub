"""Router logic: classify a question into a route via structured LLM output.

The router uses JSON-mode completion to return a validated
:class:`RouteDecision`. A deterministic heuristic fallback keeps the system
working (and tests offline) if the structured call fails.
"""

from __future__ import annotations

import re

from agentic_rag.agents.models import RouteDecision, RouteName
from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.llm import LLMError, complete_json
from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)

# Cues that strongly imply a need for fresh, real-time web information.
_WEBSEARCH_CUES = re.compile(
    r"\b(latest|today|yesterday|current|recent|news|breaking|2024|2025|2026|"
    r"price of|stock|weather|who won|right now|this week|this year)\b",
    re.IGNORECASE,
)
# Cues that imply the user refers to uploaded/indexed material.
_VECTORSTORE_CUES = re.compile(
    r"\b(document|pdf|file|paper|report|according to|in the (doc|text|paper)|"
    r"uploaded|attachment)\b",
    re.IGNORECASE,
)

_ROUTER_SYSTEM = (
    "You are a query router for a retrieval-augmented generation system. "
    "Classify the question into exactly one route: 'vectorstore' (answerable "
    "from the user's uploaded documents), 'websearch' (needs current/real-time "
    "web information), or 'generate' (general knowledge the model already has). "
    "Respond ONLY with JSON: "
    '{"route": "...", "reasoning": "...", "confidence": 0.0}.'
)


def heuristic_route(question: str, *, has_documents: bool) -> RouteDecision:
    """Classify a question without calling the LLM.

    This deterministic fallback is also what the unit tests exercise so they
    can run fully offline.

    Args:
        question: The user question.
        has_documents: Whether any documents are indexed.

    Returns:
        A :class:`RouteDecision`.
    """
    text = question.strip()
    if _WEBSEARCH_CUES.search(text):
        return RouteDecision(
            route=RouteName.WEBSEARCH,
            reasoning="Question contains time-sensitive or current-events cues.",
            confidence=0.6,
        )
    if has_documents and _VECTORSTORE_CUES.search(text):
        return RouteDecision(
            route=RouteName.VECTORSTORE,
            reasoning="Question references uploaded documents and documents exist.",
            confidence=0.6,
        )
    if has_documents:
        # Default to the documents when we have them; the grader will trigger a
        # web-search fallback if retrieval turns out to be irrelevant.
        return RouteDecision(
            route=RouteName.VECTORSTORE,
            reasoning="Documents are available; attempt grounded retrieval first.",
            confidence=0.5,
        )
    return RouteDecision(
        route=RouteName.GENERATE,
        reasoning="No documents indexed and no real-time cues detected.",
        confidence=0.5,
    )


def route_question(
    question: str,
    *,
    has_documents: bool,
    config: AppConfig | None = None,
    use_llm: bool = True,
) -> RouteDecision:
    """Route a question, preferring the LLM and falling back to heuristics.

    Args:
        question: The user question.
        has_documents: Whether any documents are indexed.
        config: Optional config override.
        use_llm: When ``False``, skip the LLM and use heuristics only.

    Returns:
        A validated :class:`RouteDecision`.
    """
    config = config or get_config()

    if not use_llm:
        return heuristic_route(question, has_documents=has_documents)

    context_hint = (
        "Documents ARE available in the vectorstore."
        if has_documents
        else "No documents are indexed; do not choose 'vectorstore'."
    )
    try:
        decision = complete_json(
            [
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": f"{context_hint}\nQuestion: {question}"},
            ],
            RouteDecision,
            config=config,
        )
        # Guard against the LLM choosing vectorstore when nothing is indexed.
        if decision.route is RouteName.VECTORSTORE and not has_documents:
            logger.info("Overriding vectorstore route: no documents indexed")
            return heuristic_route(question, has_documents=False)
        return decision
    except LLMError as exc:
        logger.warning("LLM routing failed (%s); using heuristic fallback", exc)
        return heuristic_route(question, has_documents=has_documents)
