"""Character-based recursive text chunking with overlap.

Chunking keeps semantic locality by preferring to split on paragraph, line
then sentence boundaries before falling back to a hard character cut. Each
chunk retains its source document's provenance metadata for citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag.core.config import get_config
from agentic_rag.ingestion.loader import LoadedDocument

# Separators tried in order of decreasing semantic strength.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(slots=True)
class Chunk:
    """A chunk of text ready for embedding.

    Attributes:
        text: The chunk content.
        source: Source file name.
        page: Optional page number for PDFs.
        chunk_index: Sequential index of the chunk within its document.
        file_hash: Hash of the originating file (for dedup).
    """

    text: str
    source: str
    page: int | None = None
    chunk_index: int = 0
    file_hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split ``text`` into pieces no larger than ``chunk_size`` characters.

    Tries each separator in turn; if a single piece is still too large it
    recurses with the next-weaker separator, finally hard-cutting if needed.
    """
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # No separators left — hard cut.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    parts = text.split(separator)
    pieces: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, chunk_size, rest))
    return [p for p in pieces if p]


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split a string into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Max characters per chunk (defaults to config value).
        chunk_overlap: Characters of overlap between chunks (defaults to config).

    Returns:
        A list of chunk strings. Empty input yields an empty list.
    """
    config = get_config()
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    text = (text or "").strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # First split into separator-respecting pieces, then greedily merge those
    # pieces into windows of ~chunk_size with the requested overlap.
    pieces = _split_recursive(text, chunk_size, _SEPARATORS)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # Start the next window with a tail-overlap of the previous chunk.
        overlap_tail = current[-chunk_overlap:] if chunk_overlap and current else ""
        current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece

    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a batch of loaded documents while preserving provenance.

    Args:
        documents: Loaded documents (one per PDF page or text file).
        chunk_size: Optional override for max chunk size.
        chunk_overlap: Optional override for chunk overlap.

    Returns:
        A flat list of :class:`Chunk` objects carrying source metadata.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        for index, piece in enumerate(
            chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    page=doc.page,
                    chunk_index=index,
                    file_hash=doc.file_hash,
                )
            )
    return chunks
