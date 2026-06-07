"""Tests for the text chunking logic."""

from __future__ import annotations

from agentic_rag.ingestion.chunking import chunk_documents, chunk_text
from agentic_rag.ingestion.loader import LoadedDocument


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=100, chunk_overlap=10)
    assert chunks == ["Hello world."]


def test_long_text_is_split_into_multiple_chunks():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 + 20 for c in chunks)


def test_overlap_creates_shared_content():
    text = ". ".join(f"sentence number {i}" for i in range(60))
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)
    assert len(chunks) >= 2
    # Consecutive chunks should share some overlapping tail/head content.
    overlap_found = any(
        set(chunks[i].split()) & set(chunks[i + 1].split())
        for i in range(len(chunks) - 1)
    )
    assert overlap_found


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, chunk_overlap=50)


def test_chunk_documents_preserves_provenance():
    docs = [
        LoadedDocument(text="alpha " * 200, source="a.pdf", page=1, file_hash="h1"),
        LoadedDocument(text="beta " * 200, source="b.txt", page=None, file_hash="h2"),
    ]
    chunks = chunk_documents(docs, chunk_size=150, chunk_overlap=20)
    assert chunks
    sources = {c.source for c in chunks}
    assert sources == {"a.pdf", "b.txt"}
    # chunk_index is sequential within each document.
    a_chunks = [c for c in chunks if c.source == "a.pdf"]
    assert [c.chunk_index for c in a_chunks] == list(range(len(a_chunks)))
    assert all(c.file_hash for c in chunks)
