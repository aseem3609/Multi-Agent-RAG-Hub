"""Document loaders for PDF, plain-text and Markdown files.

Each loader returns one or more :class:`LoadedDocument` objects carrying the
text plus provenance metadata (source file name and page number) used later
for citations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(slots=True)
class LoadedDocument:
    """A unit of loaded text with provenance metadata.

    Attributes:
        text: The extracted text content.
        source: The source file name (basename).
        page: 1-based page number for PDFs; ``None`` for flat text files.
        file_hash: SHA-256 of the originating file's bytes (for dedup).
    """

    text: str
    source: str
    page: int | None = None
    file_hash: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Used to skip re-embedding files that have not changed.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of in-memory bytes (e.g. an upload)."""
    return hashlib.sha256(data).hexdigest()


def _load_pdf_bytes(data: bytes, source: str, file_hash: str) -> list[LoadedDocument]:
    """Extract text per page from PDF bytes."""
    from pypdf import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    docs: list[LoadedDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            LoadedDocument(text=text, source=source, page=page_number, file_hash=file_hash)
        )
    return docs


def _load_text_bytes(data: bytes, source: str, file_hash: str) -> list[LoadedDocument]:
    """Decode text/markdown bytes into a single document."""
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    return [LoadedDocument(text=text, source=source, page=None, file_hash=file_hash)]


def load_bytes(data: bytes, filename: str) -> list[LoadedDocument]:
    """Load documents from raw bytes (e.g. a Streamlit upload).

    Args:
        data: The raw file bytes.
        filename: Original file name, used to infer type and for citations.

    Returns:
        A list of :class:`LoadedDocument` objects (one per PDF page).

    Raises:
        ValueError: If the file type is unsupported.
    """
    suffix = Path(filename).suffix.lower()
    source = Path(filename).name
    file_hash = compute_bytes_hash(data)

    if suffix == ".pdf":
        return _load_pdf_bytes(data, source, file_hash)
    if suffix in SUPPORTED_SUFFIXES:
        return _load_text_bytes(data, source, file_hash)
    raise ValueError(f"Unsupported file type: {suffix!r} ({filename})")


def load_path(path: Path) -> list[LoadedDocument]:
    """Load documents from a single file on disk.

    Args:
        path: Path to a supported document.

    Returns:
        A list of :class:`LoadedDocument` objects.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_bytes(path.read_bytes(), path.name)


def load_documents(directory: Path) -> list[LoadedDocument]:
    """Recursively load every supported document in a directory.

    Unsupported files are skipped with a warning so one bad file does not
    abort the whole batch.

    Args:
        directory: The folder to scan.

    Returns:
        A flat list of loaded documents across all files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Data directory %s does not exist; nothing to load", directory)
        return []

    documents: list[LoadedDocument] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents.extend(load_path(path))
        except Exception as exc:  # noqa: BLE001 - skip bad file, keep going
            logger.warning("Skipping %s: %s", path, exc)
    logger.info("Loaded %d document pages from %s", len(documents), directory)
    return documents
