"""Tests for the corrective-RAG grader (heuristic + LLM fallback)."""

from __future__ import annotations

from agentic_rag.agents.grader import grade_context, heuristic_grade
from agentic_rag.agents.models import GradeResult
from agentic_rag.core.llm import LLMError


def test_empty_context_is_irrelevant():
    result = grade_context("What is X?", "", use_llm=False)
    assert result.is_relevant is False
    assert result.score == 0.0


def test_relevant_context_passes_heuristic():
    question = "What is the capital of France?"
    context = "The capital of France is Paris, a major European city."
    result = heuristic_grade(question, context, threshold=0.5)
    assert result.is_relevant is True
    assert result.score >= 0.5


def test_irrelevant_context_fails_heuristic():
    question = "What is the boiling point of water?"
    context = "Penguins are flightless birds living in the southern hemisphere."
    result = heuristic_grade(question, context, threshold=0.5)
    assert result.is_relevant is False


def test_grade_context_offline_mode_uses_heuristic():
    question = "Explain neural networks"
    context = "Neural networks are layers of neurons used in machine learning."
    result = grade_context(question, context, use_llm=False)
    assert isinstance(result, GradeResult)
    assert result.is_relevant is True


def test_grade_context_uses_llm_when_available(monkeypatch):
    expected = GradeResult(is_relevant=True, score=0.95, reasoning="llm")
    monkeypatch.setattr(
        "agentic_rag.agents.grader.complete_json", lambda *a, **k: expected
    )
    result = grade_context("q", "some relevant context here", use_llm=True)
    assert result.score == 0.95
    assert result.is_relevant is True


def test_grade_context_threshold_reconciliation(monkeypatch):
    # LLM says relevant but score is below the configured threshold (0.5).
    low = GradeResult(is_relevant=True, score=0.2, reasoning="weak")
    monkeypatch.setattr(
        "agentic_rag.agents.grader.complete_json", lambda *a, **k: low
    )
    result = grade_context("q", "some context", use_llm=True)
    assert result.is_relevant is False


def test_grade_context_falls_back_on_llm_error(monkeypatch):
    def boom(*args, **kwargs):
        raise LLMError("offline")

    monkeypatch.setattr("agentic_rag.agents.grader.complete_json", boom)
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    result = grade_context(question, context, use_llm=True)
    assert result.is_relevant is True
