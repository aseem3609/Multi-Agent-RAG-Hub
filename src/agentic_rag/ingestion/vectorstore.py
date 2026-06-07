"""Persistent ChromaDB vector store wrapper.

Embeddings are generated through the provider-agnostic LLM layer
(:func:`agentic_rag.core.llm.embed_texts`) and stored in a local, persistent
ChromaDB collection. File-content hashes are tracked so unchanged files are
never re-embedded.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.llm import embed_texts
from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.chunking import Chunk

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned from a similarity search.

    Attributes:
        text: The chunk content.
        source: Source file name.
        page: Optional page number.
        score: Similarity score in ``[0, 1]`` (higher is more relevant).
    """

    text: str
    source: str
    page: int | None
    score: float

    def citation(self) -> str:
        """Return a human-readable citation label for this chunk."""
        if self.page is not None:
            return f"{self.source} (p.{self.page})"
        return self.source


class VectorStore:
    """A thin, typed wrapper around a persistent ChromaDB collection."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialise (and lazily connect to) the persistent collection.

        Args:
            config: Optional config override.
        """
        self._config = config or get_config()
        self._client = None
        self._collection = None

    # -- connection ---------------------------------------------------------
    def _ensure_collection(self):
        """Create the Chroma client/collection on first access."""
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.config import Settings

        self._config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._config.chroma_persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        # We compute embeddings ourselves, so no embedding function is attached.
        self._collection = self._client.get_or_create_collection(
            name=self._config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # -- dedup --------------------------------------------------------------
    def indexed_hashes(self) -> set[str]:
        """Return the set of file hashes already present in the collection."""
        collection = self._ensure_collection()
        try:
            records = collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001 - empty/new collection is fine
            logger.debug("Could not read existing metadatas: %s", exc)
            return set()
        return {
            meta.get("file_hash", "")
            for meta in (records.get("metadatas") or [])
            if meta and meta.get("file_hash")
        }

    def has_hash(self, file_hash: str) -> bool:
        """Return ``True`` if a file with ``file_hash`` is already indexed."""
        return file_hash in self.indexed_hashes()

    # -- writes -------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], *, skip_existing: bool = True) -> int:
        """Embed and store chunks, skipping files already indexed.

        Args:
            chunks: The chunks to add.
            skip_existing: When ``True`` (default), chunks whose ``file_hash``
                is already present are skipped to avoid re-embedding.

        Returns:
            The number of chunks actually written.
        """
        if not chunks:
            return 0

        collection = self._ensure_collection()
        existing = self.indexed_hashes() if skip_existing else set()

        pending = [c for c in chunks if not (skip_existing and c.file_hash in existing)]
        if not pending:
            logger.info("All %d chunks already indexed; nothing to do", len(chunks))
            return 0

        embeddings = embed_texts([c.text for c in pending], config=self._config)
        ids = [f"{c.file_hash}:{c.page}:{c.chunk_index}" for c in pending]
        metadatas = [
            {
                "source": c.source,
                "page": c.page if c.page is not None else -1,
                "file_hash": c.file_hash,
                "chunk_index": c.chunk_index,
            }
            for c in pending
        ]
        collection.upsert(
            ids=ids,
            documents=[c.text for c in pending],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Indexed %d new chunks", len(pending))
        return len(pending)

    # -- reads --------------------------------------------------------------
    def query(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a question.

        Args:
            question: The natural-language query.
            top_k: Number of chunks to return (defaults to config ``top_k``).

        Returns:
            A list of :class:`RetrievedChunk`, ordered by descending relevance.
        """
        collection = self._ensure_collection()
        top_k = top_k or self._config.top_k

        if collection.count() == 0:
            return []

        query_embedding = embed_texts([question], config=self._config)[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances, strict=False):
            page = meta.get("page", -1)
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", "unknown"),
                    page=None if page in (-1, None) else int(page),
                    # Cosine distance -> similarity in [0, 1].
                    score=max(0.0, 1.0 - float(distance)),
                )
            )
        return retrieved

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._ensure_collection().count()

    def reset(self) -> None:
        """Delete and recreate the collection (irreversible)."""
        collection = self._ensure_collection()
        self._client.delete_collection(collection.name)
        self._collection = None
        logger.warning("Vector store collection %r reset", self._config.chroma_collection)
