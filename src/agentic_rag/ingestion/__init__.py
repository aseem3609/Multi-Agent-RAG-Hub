"""Document ingestion: loading, chunking, embedding and vector storage."""

from __future__ import annotations

from agentic_rag.ingestion.chunking import Chunk, chunk_text, chunk_documents
from agentic_rag.ingestion.loader import LoadedDocument, load_documents, load_path
from agentic_rag.ingestion.vectorstore import VectorStore

__all__ = [
    "Chunk",
    "chunk_text",
    "chunk_documents",
    "LoadedDocument",
    "load_documents",
    "load_path",
    "VectorStore",
]
