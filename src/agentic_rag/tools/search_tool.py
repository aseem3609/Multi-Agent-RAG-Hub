"""Web search tool backed by Tavily, with graceful degradation.

If no ``TAVILY_API_KEY`` is configured, :func:`web_search` returns an empty
result set with a helpful message instead of raising, so the rest of the
pipeline keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class WebSearchHit:
    """A single web search result.

    Attributes:
        title: Result title.
        url: Result URL (used as a citation).
        content: Snippet/content extracted by the search provider.
        score: Provider relevance score.
    """

    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass(slots=True)
class WebSearchResult:
    """The outcome of a web search.

    Attributes:
        hits: The result hits (possibly empty).
        available: Whether the search backend was usable.
        message: Human-readable status (e.g. why search was unavailable).
    """

    hits: list[WebSearchHit] = field(default_factory=list)
    available: bool = True
    message: str = ""


def web_search(
    query: str,
    *,
    max_results: int = 5,
    config: AppConfig | None = None,
) -> WebSearchResult:
    """Search the web for ``query`` using Tavily.

    Args:
        query: The search query.
        max_results: Maximum number of hits to return.
        config: Optional config override.

    Returns:
        A :class:`WebSearchResult`. When Tavily is not configured or errors,
        ``available`` is ``False`` and ``hits`` is empty.
    """
    config = config or get_config()

    if not config.web_search_enabled:
        msg = (
            "Web search is unavailable: set TAVILY_API_KEY in your environment "
            "to enable live web results."
        )
        logger.warning(msg)
        return WebSearchResult(hits=[], available=False, message=msg)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )
        hits = [
            WebSearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in response.get("results", [])
        ]
        logger.info("Tavily returned %d results for %r", len(hits), query)
        return WebSearchResult(hits=hits, available=True, message="")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any error
        msg = f"Web search failed: {exc}"
        logger.error(msg)
        return WebSearchResult(hits=[], available=False, message=msg)


def format_web_context(result: WebSearchResult) -> str:
    """Render web hits into a context block for the LLM prompt."""
    if not result.hits:
        return ""
    blocks = [
        f"[{i}] {hit.title}\nURL: {hit.url}\n{hit.content}"
        for i, hit in enumerate(result.hits, start=1)
    ]
    return "\n\n".join(blocks)
