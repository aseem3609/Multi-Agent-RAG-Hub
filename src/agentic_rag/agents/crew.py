"""The Agentic RAG orchestrator (the "crew").

This module wires the router, retriever, grader and generators into a single
``answer`` flow implementing the corrective-RAG pattern:

    route → (retrieve → grade → maybe fall back to web) → answer + citations

The deterministic Python flow is the source of truth (reliable + testable).
CrewAI agents are also constructed from the YAML configs so the system stays
aligned with CrewAI conventions and can be extended with autonomous behaviour.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_rag.agents.grader import grade_context
from agentic_rag.agents.models import Citation, RagAnswer, RouteDecision, RouteName
from agentic_rag.agents.router import route_question
from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.vectorstore import VectorStore
from agentic_rag.tools.generation_tool import generate_answer, generate_grounded_answer
from agentic_rag.tools.rag_tool import format_context, retrieve_context
from agentic_rag.tools.search_tool import format_web_context, web_search

logger = get_logger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"


def load_yaml_config(name: str) -> dict:
    """Load a YAML config file from the ``config`` directory.

    Args:
        name: File name (e.g. ``"agents.yaml"``).

    Returns:
        The parsed YAML as a dict (empty dict if the file is missing).
    """
    path = _CONFIG_DIR / name
    if not path.is_file():
        logger.warning("YAML config %s not found", path)
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class AgenticRAGCrew:
    """Orchestrates routing, retrieval, grading and answer generation."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        store: VectorStore | None = None,
        use_llm: bool = True,
    ) -> None:
        """Initialise the crew.

        Args:
            config: Optional config override.
            store: Optional vector store override (useful for tests).
            use_llm: When ``False``, routing/grading use offline heuristics.
        """
        self.config = config or get_config()
        self.store = store or VectorStore(self.config)
        self.use_llm = use_llm
        self.agents_config = load_yaml_config("agents.yaml")
        self.tasks_config = load_yaml_config("tasks.yaml")

    # -- helpers ------------------------------------------------------------
    def _has_documents(self) -> bool:
        """Return whether any chunks are indexed (errs on the side of False)."""
        try:
            return self.store.count() > 0
        except Exception as exc:  # noqa: BLE001 - treat store errors as empty
            logger.warning("Could not read vector store count: %s", exc)
            return False

    # -- route handlers -----------------------------------------------------
    def _answer_from_vectorstore(
        self, question: str, history: list[dict[str, str]] | None
    ) -> RagAnswer:
        """Retrieve, grade and answer from documents; fall back to web on miss."""
        chunks = retrieve_context(question, store=self.store, config=self.config)
        context = format_context(chunks)
        grade = grade_context(question, context, config=self.config, use_llm=self.use_llm)
        logger.info("Grade: relevant=%s score=%.2f", grade.is_relevant, grade.score)

        if not grade.is_relevant:
            # Corrective-RAG: documents were insufficient -> try the web.
            logger.info("Context insufficient; falling back to web search")
            web = self._answer_from_web(question, history)
            web.used_fallback = True
            return web

        answer = generate_grounded_answer(
            question, context, history=history, config=self.config
        )
        citations = [Citation(label=chunk.citation()) for chunk in chunks]
        return RagAnswer(
            answer=answer,
            route=RouteName.VECTORSTORE,
            citations=_dedupe_citations(citations),
        )

    def _answer_from_web(
        self, question: str, history: list[dict[str, str]] | None
    ) -> RagAnswer:
        """Search the web and answer from the results."""
        result = web_search(question, config=self.config)
        if not result.available or not result.hits:
            return RagAnswer(
                answer=result.message
                or "I couldn't find relevant web results for that question.",
                route=RouteName.WEBSEARCH,
                citations=[],
            )
        context = format_web_context(result)
        answer = generate_grounded_answer(
            question, context, history=history, config=self.config
        )
        citations = [Citation(label=hit.title or hit.url, url=hit.url) for hit in result.hits]
        return RagAnswer(
            answer=answer,
            route=RouteName.WEBSEARCH,
            citations=_dedupe_citations(citations),
        )

    def _answer_from_generation(
        self, question: str, history: list[dict[str, str]] | None
    ) -> RagAnswer:
        """Answer from the LLM's own knowledge."""
        answer = generate_answer(question, history=history, config=self.config)
        return RagAnswer(answer=answer, route=RouteName.GENERATE, citations=[])

    # -- public API ---------------------------------------------------------
    def route(self, question: str) -> RouteDecision:
        """Return the routing decision for a question (without answering)."""
        return route_question(
            question,
            has_documents=self._has_documents(),
            config=self.config,
            use_llm=self.use_llm,
        )

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> RagAnswer:
        """Answer a question end-to-end through the agentic flow.

        Args:
            question: The user question.
            history: Optional short chat history for follow-up context.

        Returns:
            A :class:`RagAnswer` with answer text, route and citations.
        """
        question = (question or "").strip()
        if not question:
            return RagAnswer(
                answer="Please provide a question.",
                route=RouteName.GENERATE,
                citations=[],
            )

        decision = self.route(question)
        logger.info("Routed to %s (%.2f): %s", decision.route, decision.confidence, decision.reasoning)

        try:
            if decision.route is RouteName.VECTORSTORE:
                return self._answer_from_vectorstore(question, history)
            if decision.route is RouteName.WEBSEARCH:
                return self._answer_from_web(question, history)
            return self._answer_from_generation(question, history)
        except Exception as exc:  # noqa: BLE001 - never crash the caller
            logger.error("Answering failed: %s", exc)
            return RagAnswer(
                answer=f"Sorry, an error occurred while answering: {exc}",
                route=decision.route,
                citations=[],
            )


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """Remove duplicate citations while preserving order."""
    seen: set[str] = set()
    unique: list[Citation] = []
    for citation in citations:
        key = citation.url or citation.label
        if key and key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique
