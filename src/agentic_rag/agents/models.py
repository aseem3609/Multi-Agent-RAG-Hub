"""Pydantic models for structured agent inputs and outputs.

Using explicit models (rather than loose strings) makes the routing,
grading and answer steps reliable, validatable and easy to test.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RouteName(str, Enum):
    """The three routes a question can be dispatched to."""

    VECTORSTORE = "vectorstore"
    WEBSEARCH = "websearch"
    GENERATE = "generate"


class RouteDecision(BaseModel):
    """The router agent's structured decision for a question.

    Attributes:
        route: Which subsystem should answer the question.
        reasoning: A short justification (useful for logging/debugging).
        confidence: Router confidence in ``[0, 1]``.
    """

    route: RouteName = Field(description="Chosen route for the question.")
    reasoning: str = Field(default="", description="Why this route was chosen.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GradeResult(BaseModel):
    """The grader's verdict on whether retrieved context answers a question.

    Attributes:
        is_relevant: Whether the context sufficiently answers the question.
        score: Relevance score in ``[0, 1]``.
        reasoning: Short justification.
    """

    is_relevant: bool = Field(description="True if context answers the question.")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class Citation(BaseModel):
    """A single source citation backing an answer.

    Attributes:
        label: Human-readable citation (file + page, or page title).
        url: Optional source URL for web results.
    """

    label: str
    url: str | None = None


class RagAnswer(BaseModel):
    """The final, user-facing answer with provenance.

    Attributes:
        answer: The natural-language answer (or an honest "I don't know").
        route: The route that produced the answer.
        citations: Supporting citations.
        used_fallback: Whether corrective web-search fallback was triggered.
    """

    answer: str
    route: RouteName
    citations: list[Citation] = Field(default_factory=list)
    used_fallback: bool = False
