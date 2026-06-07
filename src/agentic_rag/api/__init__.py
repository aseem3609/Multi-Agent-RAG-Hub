"""FastAPI application exposing the Agentic RAG system over HTTP."""

from __future__ import annotations

from agentic_rag.api.main import app, run

__all__ = ["app", "run"]
