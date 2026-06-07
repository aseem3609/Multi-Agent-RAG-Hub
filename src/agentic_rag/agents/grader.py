"""Corrective-RAG grader: decide whether retrieved context is good enough.

After retrieval, the grader scores how well the context answers the question.
If the score falls below the configured threshold, the orchestrator falls back
to web search (the CRAG pattern). A deterministic lexical-overlap fallback
keeps the grader working offline.
"""

from __future__ import annotations

import re

from agentic_rag.agents.models import GradeResult
from agentic_rag.core.config import AppConfig, get_config
from agentic_rag.core.llm import LLMError, complete_json
from agentic_rag.core.logging import get_logger

logger = get_logger(__name__)

_GRADER_SYSTEM = (
    "You grade whether a context passage answers a question for a "
    "corrective-RAG system. Be strict: only mark relevant if the context "
    "actually contains the information needed. Respond ONLY with JSON: "
    '{"is_relevant": true/false, "score": 0.0, "reasoning": "..."}.'
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "is", "are", "and", "or",
    "what", "who", "when", "where", "why", "how", "does", "do", "did",
    "for", "with", "about", "this", "that", "it", "as", "by", "be",
}


def heuristic_grade(question: str, context: str, *, threshold: float) -> GradeResult:
    """Grade context relevance via lexical token overlap (no LLM).

    Args:
        question: The user question.
        context: The retrieved context block.
        threshold: Minimum overlap ratio to be considered relevant.

    Returns:
        A :class:`GradeResult`.
    """
    if not context.strip():
        return GradeResult(is_relevant=False, score=0.0, reasoning="Empty context.")

    q_tokens = {w for w in _WORD_RE.findall(question.lower()) if w not in _STOPWORDS}
    if not q_tokens:
        return GradeResult(is_relevant=True, score=1.0, reasoning="No content words to check.")

    c_tokens = set(_WORD_RE.findall(context.lower()))
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    return GradeResult(
        is_relevant=overlap >= threshold,
        score=round(overlap, 3),
        reasoning=f"Lexical overlap {overlap:.2f} vs threshold {threshold:.2f}.",
    )


def grade_context(
    question: str,
    context: str,
    *,
    config: AppConfig | None = None,
    use_llm: bool = True,
) -> GradeResult:
    """Grade whether ``context`` answers ``question``.

    Args:
        question: The user question.
        context: The retrieved context block.
        config: Optional config override.
        use_llm: When ``False``, use the lexical heuristic only.

    Returns:
        A validated :class:`GradeResult`.
    """
    config = config or get_config()

    if not context.strip():
        return GradeResult(is_relevant=False, score=0.0, reasoning="Empty context.")

    if not use_llm:
        return heuristic_grade(question, context, threshold=config.grade_threshold)

    try:
        result = complete_json(
            [
                {"role": "system", "content": _GRADER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
            GradeResult,
            config=config,
        )
        # Reconcile the boolean flag with the configured numeric threshold.
        result.is_relevant = result.is_relevant and result.score >= config.grade_threshold
        return result
    except LLMError as exc:
        logger.warning("LLM grading failed (%s); using heuristic fallback", exc)
        return heuristic_grade(question, context, threshold=config.grade_threshold)
