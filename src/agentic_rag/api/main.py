"""FastAPI application exposing the Agentic RAG pipeline.

Endpoints:
    GET  /health   : liveness probe.
    GET  /stats    : vector-store statistics.
    POST /ask      : answer a question through the agentic flow.
    POST /ingest   : upload and index one or more documents.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agentic_rag.agents.crew import AgenticRAGCrew
from agentic_rag.agents.models import Citation, RouteName
from agentic_rag.core.config import get_config
from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.ingestion.pipeline import ingest_upload
from agentic_rag.ingestion.vectorstore import VectorStore

logger = get_logger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    version="0.1.0",
    description="Multi-agent Retrieval-Augmented Generation with corrective fallback.",
)


# --- request/response schemas ---------------------------------------------
class ChatTurn(BaseModel):
    """A single conversation turn."""

    role: str = Field(description="'user' or 'assistant'.")
    content: str


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=1, description="The user's question.")
    history: list[ChatTurn] = Field(default_factory=list, description="Prior turns.")


class AskResponse(BaseModel):
    """Response body for ``POST /ask``."""

    answer: str
    route: RouteName
    citations: list[Citation]
    used_fallback: bool


class IngestResponse(BaseModel):
    """Response body for ``POST /ingest``."""

    files: int
    chunks_indexed: int
    skipped_unchanged: bool


# --- dependencies ----------------------------------------------------------
def get_crew() -> AgenticRAGCrew:
    """Construct a crew instance (kept simple; cheap to build)."""
    return AgenticRAGCrew()


@app.on_event("startup")
def _on_startup() -> None:
    """Configure logging when the API process starts."""
    configure_logging(get_config().log_level)
    logger.info("Agentic RAG API started")


# --- routes ----------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict[str, object]:
    """Return vector-store statistics and provider configuration."""
    config = get_config()
    try:
        count = VectorStore(config).count()
    except Exception as exc:  # noqa: BLE001 - report rather than crash
        logger.warning("Could not read store count: %s", exc)
        count = 0
    return {
        "provider": config.llm_provider.value,
        "model": config.llm_model,
        "indexed_chunks": count,
        "web_search_enabled": config.web_search_enabled,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question through the agentic routing + corrective-RAG flow."""
    crew = get_crew()
    history = [turn.model_dump() for turn in request.history]
    result = crew.answer(request.question, history=history)
    return AskResponse(
        answer=result.answer,
        route=result.route,
        citations=result.citations,
        used_fallback=result.used_fallback,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: Annotated[list[UploadFile], File()]) -> IngestResponse:
    """Upload and index one or more documents (PDF/TXT/MD)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    store = VectorStore()
    total_files = 0
    total_chunks = 0
    all_unchanged = True
    for upload in files:
        try:
            data = await upload.read()
            result = ingest_upload(data, upload.filename or "upload", store=store)
            total_files += result.files
            total_chunks += result.chunks_indexed
            all_unchanged = all_unchanged and result.skipped_unchanged
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface as 500 with message
            logger.error("Ingestion failed for %s: %s", upload.filename, exc)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        files=total_files,
        chunks_indexed=total_chunks,
        skipped_unchanged=all_unchanged and total_chunks == 0,
    )


def run() -> None:
    """Run the API with uvicorn (``agentic-rag-api`` console script)."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "agentic_rag.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
