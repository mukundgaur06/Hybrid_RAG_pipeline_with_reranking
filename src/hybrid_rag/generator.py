"""Grounded answer generator with citation attribution and safe offline fallback.

Uses the official google-genai SDK to generate strictly grounded answers
from retrieved evidence passages, assigns explicit source identifiers ([S1], [S2]...),
validates citation references, and falls back gracefully to an Evidence Viewer mode
when no API key is present or when API requests encounter errors.
"""

import os
import re
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

from hybrid_rag.schema import (
    Chunk,
    PipelineResponse,
    RerankedResult,
    RetrievalResult,
    SourcePassage,
)

CITATION_PATTERN = re.compile(r"\[S\d+\]")


class GroundedAnswerGenerator:
    """Generates strictly grounded answers from retrieved context using Gemini."""

    DEFAULT_MODEL = "gemini-3.8-flash"
    # Allow repository-wide override via environment variable
    _ENV_MODEL_VAR = "GEMINI_MODEL"
    INSUFFICIENT_EVIDENCE_PHRASE = (
        "The provided evidence does not contain sufficient information to answer the question."
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        """Initialize the grounded answer generator.

        Args:
            model_name: Gemini model identifier (e.g., 'gemini-3.8-flash').
                If ``None``, the value of the ``GEMINI_MODEL`` environment
                variable is used, falling back to ``DEFAULT_MODEL``.
            api_key: Optional API key. If omitted, reads from GEMINI_API_KEY or GOOGLE_API_KEY.
            client: Optional injected genai.Client test double for offline testing.
        """
        self.model_name = (
            model_name
            or os.environ.get(self._ENV_MODEL_VAR)
            or self.DEFAULT_MODEL
        )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = client

    def _get_client(self) -> Any:
        """Lazily initialize the google-genai Client if not injected."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("No Gemini API key configured.")
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def has_api_key(self) -> bool:
        """Check if an API key or injected client is available."""
        return bool(self._client is not None or self._api_key)

    def format_sources(
        self, candidates: Sequence[Union[RerankedResult, RetrievalResult, Chunk]]
    ) -> List[SourcePassage]:
        """Assign stable [S1], [S2], ... identifiers to candidate passages."""
        sources: List[SourcePassage] = []
        for idx, item in enumerate(candidates, start=1):
            source_id = f"[S{idx}]"
            if isinstance(item, RerankedResult):
                sources.append(
                    SourcePassage(source_id=source_id, chunk=item.chunk, reranked_result=item)
                )
            elif isinstance(item, RetrievalResult):
                sources.append(SourcePassage(source_id=source_id, chunk=item.chunk))
            elif isinstance(item, Chunk):
                sources.append(SourcePassage(source_id=source_id, chunk=item))
            else:
                chunk = getattr(item, "chunk", None)
                if isinstance(chunk, Chunk):
                    sources.append(SourcePassage(source_id=source_id, chunk=chunk))
        return sources

    def build_context_string(self, sources: Sequence[SourcePassage]) -> str:
        """Construct the textual context block from assigned sources."""
        context_parts: List[str] = []
        for src in sources:
            page_info = (
                f", Page {src.chunk.page_number}"
                if src.chunk.page_number is not None
                else ""
            )
            header = f"Source {src.source_id} (File: {src.chunk.document_name}{page_info}):"
            context_parts.append(f"{header}\n{src.chunk.text.strip()}\n")
        return "\n".join(context_parts)

    def build_prompt(self, query: str, context_str: str) -> str:
        """Construct a strict grounding prompt with citation and refusal guidelines."""
        return (
            "You are a factual, strictly grounded question-answering assistant.\n"
            "Your task is to answer the user's question using ONLY the provided evidence passages.\n\n"
            "RULES:\n"
            "1. Answer ONLY based on the facts directly stated in the evidence below.\n"
            "2. Do NOT extrapolate, speculate, or incorporate outside knowledge.\n"
            "3. Cite every factual claim using the exact source identifiers in brackets, e.g. [S1] or [S2].\n"
            f'4. If the provided evidence is insufficient to answer the question, respond with: "{self.INSUFFICIENT_EVIDENCE_PHRASE}"\n\n'
            f"=== EVIDENCE PASSAGES ===\n"
            f"{context_str}\n"
            f"=== END OF EVIDENCE ===\n\n"
            f"User Question: {query}\n\n"
            "Grounded Answer (with citations):"
        )

    def validate_citations(
        self, answer_text: str, valid_source_ids: Set[str]
    ) -> Tuple[List[str], List[str]]:
        """Extract and validate citations present in the generated answer text.

        Returns:
            Tuple of (valid_citations, unvalidated_citations).
        """
        found_tags = CITATION_PATTERN.findall(answer_text)
        # Deduplicate while preserving order of appearance
        unique_tags = list(dict.fromkeys(found_tags))

        valid: List[str] = []
        unvalidated: List[str] = []

        for tag in unique_tags:
            if tag in valid_source_ids:
                valid.append(tag)
            else:
                unvalidated.append(tag)

        return valid, unvalidated

    def generate(
        self,
        query: str,
        candidates: Sequence[Union[RerankedResult, RetrievalResult, Chunk]],
    ) -> PipelineResponse:
        """Generate a grounded answer, or return an evidence-only fallback.

        Args:
            query: The user query string.
            candidates: Candidate passages (typically top-M RerankedResults).

        Returns:
            A structured PipelineResponse containing the answer, sources, and citation validity.
        """
        # Handle empty query
        if not query or not query.strip():
            return PipelineResponse(
                query=query or "",
                answer="No query was provided.",
                sources=[],
                is_fallback=False,
            )

        # Handle empty candidates
        if not candidates:
            return PipelineResponse(
                query=query,
                answer="No relevant documents or evidence found to answer this question.",
                sources=[],
                is_fallback=False,
            )

        # Assign stable source identifiers: [S1], [S2]...
        sources = self.format_sources(candidates)
        valid_source_ids = {s.source_id for s in sources}
        context_str = self.build_context_string(sources)

        # Check for missing API key / client -> activate safe Evidence Viewer fallback
        if not self.has_api_key:
            return PipelineResponse(
                query=query,
                answer=(
                    "[Evidence Viewer Mode] No Gemini API key was configured. "
                    "LLM answer synthesis was skipped. Inspect the retrieved evidence passages below."
                ),
                sources=sources,
                citations=[],
                unvalidated_citations=[],
                is_fallback=True,
                fallback_reason="No Gemini API key configured. Displaying evidence-only mode.",
                model_name=None,
            )

        # Construct strictly grounded prompt
        prompt = self.build_prompt(query=query, context_str=context_str)

        # Execute LLM call with exception safety
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            answer_text = response.text or ""
            valid_citations, unvalidated_citations = self.validate_citations(
                answer_text, valid_source_ids
            )

            return PipelineResponse(
                query=query,
                answer=answer_text,
                sources=sources,
                citations=valid_citations,
                unvalidated_citations=unvalidated_citations,
                is_fallback=False,
                fallback_reason=None,
                model_name=self.model_name,
            )

        except Exception as exc:
            # Fallback on API failure without exposing secrets
            error_msg = f"{type(exc).__name__}: {str(exc)}"
            # Strip potential key traces if present in raw exception string
            if self._api_key and self._api_key in error_msg:
                error_msg = error_msg.replace(self._api_key, "[REDACTED_API_KEY]")

            return PipelineResponse(
                query=query,
                answer=(
                    f"[Evidence Viewer Mode] LLM generation failed ({type(exc).__name__}). "
                    "Displaying retrieved evidence passages below."
                ),
                sources=sources,
                citations=[],
                unvalidated_citations=[],
                is_fallback=True,
                fallback_reason=f"LLM API call failed: {error_msg}",
                model_name=self.model_name,
            )
