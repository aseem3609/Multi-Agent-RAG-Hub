"""High-level ingestion pipeline orchestrating load → chunk → embed → store.

Provides both a programmatic API (used by the UI/API) and a small CLI entry
point (``agentic-rag-ingest``) for indexing the ``data/`` directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.ingestion.chunking import chunk_documents
from agentic_rag.ingestion.loader import load_bytes, load_documents, load_path
from agentic_rag.ingestion.vectorstore import VectorStore

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Summary of an ingestion run.

    Attributes:
        files: Number of distinct source files processed.
        chunks_indexed: Number of new chunks written to the vector store.
        skipped_unchanged: ``True`` if everything was already indexed.
    """

    files: int
    chunks_indexed: int
    skipped_unchanged: bool


def ingest_documents(documents, store: VectorStore) -> IngestionResult:
    """Chunk and index a list of already-loaded documents.

    Args:
        documents: Loaded documents to index.
        store: The vector store to write to.

    Returns:
        An :class:`IngestionResult` summary.
    """
    sources = {doc.source for doc in documents}
    chunks = chunk_documents(documents)
    indexed = store.add_chunks(chunks, skip_existing=True)
    return IngestionResult(
        files=len(sources),
        chunks_indexed=indexed,
        skipped_unchanged=indexed == 0 and bool(chunks),
    )


def ingest_directory(
    directory: Path | None = None,
    *,
    config: AppConfig | None = None,
    store: VectorStore | None = None,
) -> IngestionResult:
    """Load, chunk and index every supported file in a directory.

    Args:
        directory: Folder to ingest (defaults to config ``data_dir``).
        config: Optional config override.
        store: Optional vector store override (useful for tests).

    Returns:
        An :class:`IngestionResult` summary.
    """
    config = config or get_config()
    directory = directory or config.data_dir
    store = store or VectorStore(config)

    documents = load_documents(Path(directory))
    if not documents:
        return IngestionResult(files=0, chunks_indexed=0, skipped_unchanged=False)
    return ingest_documents(documents, store)


def ingest_file(path: Path, *, store: VectorStore | None = None) -> IngestionResult:
    """Load, chunk and index a single file on disk."""
    store = store or VectorStore()
    documents = load_path(Path(path))
    return ingest_documents(documents, store)


def ingest_upload(
    data: bytes, filename: str, *, store: VectorStore | None = None
) -> IngestionResult:
    """Load, chunk and index an in-memory upload (e.g. from Streamlit).

    Args:
        data: Raw file bytes.
        filename: Original file name.
        store: Optional vector store override.

    Returns:
        An :class:`IngestionResult` summary.
    """
    store = store or VectorStore()
    documents = load_bytes(data, filename)
    return ingest_documents(documents, store)


def cli() -> None:
    """CLI entry point: ``agentic-rag-ingest [--dir PATH]``."""
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory to ingest (defaults to DATA_DIR from config).",
    )
    args = parser.parse_args()

    config = get_config()
    configure_logging(config.log_level)
    result = ingest_directory(args.dir, config=config)
    logger.info(
        "Ingestion complete: files=%d chunks_indexed=%d unchanged=%s",
        result.files,
        result.chunks_indexed,
        result.skipped_unchanged,
    )


if __name__ == "__main__":
    cli()
