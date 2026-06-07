"""CrewAI-compatible tools and standalone helpers for retrieval/search/gen."""

from __future__ import annotations

from agentic_rag.tools.generation_tool import generate_answer
from agentic_rag.tools.rag_tool import format_context, retrieve_context
from agentic_rag.tools.search_tool import WebSearchResult, web_search

__all__ = [
    "generate_answer",
    "format_context",
    "retrieve_context",
    "WebSearchResult",
    "web_search",
]
