"""Agentic RAG — a multi-agent Retrieval-Augmented Generation system.

The package is organised into the following sub-packages:

- ``core``      : configuration, logging and the provider-agnostic LLM layer.
- ``ingestion`` : document loading, chunking, embedding and vector storage.
- ``tools``     : CrewAI tools (RAG retrieval, web search, generation).
- ``agents``    : router/grader logic, Pydantic models and the CrewAI crew.
- ``api``       : FastAPI application exposing ``POST /ask``.
- ``ui``        : Streamlit chat application.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
