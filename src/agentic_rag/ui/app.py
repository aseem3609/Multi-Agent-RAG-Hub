"""Streamlit chat application for the Agentic RAG system.

Run with:
    streamlit run src/agentic_rag/ui/app.py

Features:
    - Upload PDFs/TXT/MD and index them into ChromaDB.
    - Chat interface with short-term conversation memory.
    - Each answer shows its route (vectorstore / websearch / generate),
      whether corrective fallback fired, and clickable source citations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/agentic_rag/ui/app.py` without an editable install
# by ensuring the `src` directory is importable.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st  # noqa: E402

from agentic_rag.agents.crew import AgenticRAGCrew  # noqa: E402
from agentic_rag.core.config import get_config  # noqa: E402
from agentic_rag.core.logging import configure_logging  # noqa: E402
from agentic_rag.ingestion.pipeline import ingest_upload  # noqa: E402
from agentic_rag.ingestion.vectorstore import VectorStore  # noqa: E402

_ROUTE_BADGES = {
    "vectorstore": "📄 Documents",
    "websearch": "🌐 Web search",
    "generate": "🧠 Model knowledge",
}


@st.cache_resource(show_spinner=False)
def _get_crew() -> AgenticRAGCrew:
    """Build (and cache) a single crew instance for the session."""
    configure_logging(get_config().log_level)
    return AgenticRAGCrew()


@st.cache_resource(show_spinner=False)
def _get_store() -> VectorStore:
    """Build (and cache) a single vector store instance."""
    return VectorStore()


def _render_sidebar() -> None:
    """Render the document-upload sidebar and indexing controls."""
    config = get_config()
    store = _get_store()

    with st.sidebar:
        st.header("📚 Knowledge base")
        st.caption(
            f"Provider: **{config.llm_provider.value}** · Model: **{config.llm_model}**"
        )
        try:
            st.metric("Indexed chunks", store.count())
        except Exception as exc:  # noqa: BLE001 - show error, keep UI alive
            st.warning(f"Vector store unavailable: {exc}")

        if not config.web_search_enabled:
            st.info("Web search disabled — set `TAVILY_API_KEY` to enable it.")

        uploads = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Index documents", type="primary"):
            with st.spinner("Embedding and indexing…"):
                total = 0
                for upload in uploads:
                    try:
                        result = ingest_upload(upload.getvalue(), upload.name, store=store)
                        total += result.chunks_indexed
                    except Exception as exc:  # noqa: BLE001 - report per-file
                        st.error(f"Failed to index {upload.name}: {exc}")
                st.success(f"Indexed {total} new chunk(s).")
                st.rerun()


def _render_message(message: dict) -> None:
    """Render a single chat message, including citations metadata."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if not meta:
            return
        badge = _ROUTE_BADGES.get(meta["route"], meta["route"])
        suffix = " · corrective fallback ↩️" if meta.get("used_fallback") else ""
        st.caption(f"Answered via {badge}{suffix}")
        if meta.get("citations"):
            with st.expander("Sources"):
                for citation in meta["citations"]:
                    if citation.get("url"):
                        st.markdown(f"- [{citation['label']}]({citation['url']})")
                    else:
                        st.markdown(f"- {citation['label']}")


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
    st.title("🤖 Agentic RAG")
    st.caption(
        "Ask about your documents, the live web, or general knowledge — "
        "an agent routes each question automatically."
    )

    _render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Ask a question…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_message({"role": "user", "content": prompt})

    crew = _get_crew()
    # Pass short-term history (exclude the just-added prompt) for follow-ups.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        result = crew.answer(prompt, history=history)
        meta = {
            "route": result.route.value,
            "used_fallback": result.used_fallback,
            "citations": [c.model_dump() for c in result.citations],
        }
        message = {"role": "assistant", "content": result.answer, "meta": meta}

    st.session_state.messages.append(message)
    _render_message(message)


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the module rather than running __main__.
    main()
