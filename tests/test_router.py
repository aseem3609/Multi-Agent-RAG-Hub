"""Tests for the router logic (heuristic + LLM fallback), fully offline."""

from __future__ import annotations

from agentic_rag.agents.models import RouteDecision, RouteName
from agentic_rag.agents.router import heuristic_route, route_question
from agentic_rag.core.llm import LLMError


def test_websearch_cue_routes_to_websearch():
    decision = heuristic_route("What is the latest news on AI today?", has_documents=True)
    assert decision.route is RouteName.WEBSEARCH


def test_document_reference_routes_to_vectorstore_when_docs_present():
    decision = heuristic_route("What does the uploaded document say about X?", has_documents=True)
    assert decision.route is RouteName.VECTORSTORE


def test_defaults_to_vectorstore_when_documents_present():
    decision = heuristic_route("Explain the methodology.", has_documents=True)
    assert decision.route is RouteName.VECTORSTORE


def test_defaults_to_generate_without_documents():
    decision = heuristic_route("What is the capital of France?", has_documents=False)
    assert decision.route is RouteName.GENERATE


def test_recent_year_triggers_websearch():
    decision = heuristic_route("Who won the 2026 election?", has_documents=False)
    assert decision.route is RouteName.WEBSEARCH


def test_route_question_uses_llm_when_available(monkeypatch):
    expected = RouteDecision(route=RouteName.GENERATE, reasoning="llm", confidence=0.9)

    def fake_complete_json(messages, schema, *, config=None):
        return expected

    monkeypatch.setattr("agentic_rag.agents.router.complete_json", fake_complete_json)
    decision = route_question("Anything", has_documents=False, use_llm=True)
    assert decision is expected


def test_route_question_overrides_vectorstore_without_docs(monkeypatch):
    # Even if the LLM picks vectorstore, no docs means we must not route there.
    bad = RouteDecision(route=RouteName.VECTORSTORE, reasoning="llm", confidence=0.9)
    monkeypatch.setattr(
        "agentic_rag.agents.router.complete_json", lambda *a, **k: bad
    )
    decision = route_question("Anything", has_documents=False, use_llm=True)
    assert decision.route is not RouteName.VECTORSTORE


def test_route_question_falls_back_on_llm_error(monkeypatch):
    def boom(*args, **kwargs):
        raise LLMError("network down")

    monkeypatch.setattr("agentic_rag.agents.router.complete_json", boom)
    decision = route_question("latest news", has_documents=False, use_llm=True)
    # Heuristic still classifies the time-sensitive query correctly.
    assert decision.route is RouteName.WEBSEARCH


def test_offline_mode_skips_llm():
    decision = route_question("Explain transformers", has_documents=False, use_llm=False)
    assert decision.route is RouteName.GENERATE
