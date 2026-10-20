"""Unit tests for GroundedAnswerGenerator.

Uses an injected test double (FakeGenAIClient) for 100% offline, deterministic
testing of prompt formatting, citation validation, missing-key fallback,
and API exception handling without network requests or API keys.
"""

from typing import Any, List, Optional
import pytest

from hybrid_rag.generator import GroundedAnswerGenerator
from hybrid_rag.schema import Chunk, RerankedResult


class FakeGenAIResponse:
    """Test double representing a Gemini generate_content response."""

    def __init__(self, text: str):
        self.text = text


class FakeGenAIModels:
    """Test double for client.models."""

    def __init__(self, response_text: str = "", raise_error: Optional[Exception] = None):
        self.response_text = response_text
        self.raise_error = raise_error
        self.last_prompt: Optional[str] = None
        self.last_model: Optional[str] = None

    def generate_content(self, model: str, contents: str, **kwargs: Any) -> FakeGenAIResponse:
        self.last_model = model
        self.last_prompt = contents
        if self.raise_error:
            raise self.raise_error
        return FakeGenAIResponse(text=self.response_text)


class FakeGenAIClient:
    """Test double for genai.Client."""

    def __init__(self, response_text: str = "", raise_error: Optional[Exception] = None):
        self.models = FakeGenAIModels(response_text=response_text, raise_error=raise_error)


def _create_sample_reranked_results() -> List[RerankedResult]:
    c1 = Chunk(
        chunk_id="chunk_1",
        document_id="doc1",
        document_name="apollo.md",
        page_number=1,
        chunk_index=0,
        text="Saturn V rocket was engineered under Wernher von Braun.",
        char_start=0,
        char_end=55,
    )
    c2 = Chunk(
        chunk_id="chunk_2",
        document_id="doc2",
        document_name="space.txt",
        page_number=None,
        chunk_index=0,
        text="Apollo 11 landed on the Sea of Tranquility.",
        char_start=0,
        char_end=43,
    )
    r1 = RerankedResult(chunk=c1, reranker_score=4.5, rerank_rank=1, rrf_score=0.03, rrf_rank=1)
    r2 = RerankedResult(chunk=c2, reranker_score=3.2, rerank_rank=2, rrf_score=0.02, rrf_rank=2)
    return [r1, r2]


def test_format_sources_assigns_stable_ids():
    """Verify that candidate chunks receive deterministic [S1], [S2] source identifiers."""
    generator = GroundedAnswerGenerator()
    candidates = _create_sample_reranked_results()

    sources = generator.format_sources(candidates)

    assert len(sources) == 2
    assert sources[0].source_id == "[S1]"
    assert sources[0].chunk.chunk_id == "chunk_1"
    assert sources[0].reranked_result == candidates[0]

    assert sources[1].source_id == "[S2]"
    assert sources[1].chunk.chunk_id == "chunk_2"
    assert sources[1].reranked_result == candidates[1]


def test_build_context_and_prompt_formatting():
    """Verify context includes document names, pages, and strict grounding instructions."""
    generator = GroundedAnswerGenerator()
    candidates = _create_sample_reranked_results()
    sources = generator.format_sources(candidates)

    context_str = generator.build_context_string(sources)
    assert "Source [S1] (File: apollo.md, Page 1):" in context_str
    assert "Saturn V rocket was engineered under Wernher von Braun." in context_str
    assert "Source [S2] (File: space.txt):" in context_str
    assert "Apollo 11 landed on the Sea of Tranquility." in context_str

    prompt = generator.build_prompt("What rocket was used?", context_str)
    assert "User Question: What rocket was used?" in prompt
    assert "Answer ONLY based on the facts directly stated in the evidence below." in prompt
    assert "Cite every factual claim using the exact source identifiers in brackets" in prompt
    assert generator.INSUFFICIENT_EVIDENCE_PHRASE in prompt


def test_citation_extraction_and_validation():
    """Verify that cited IDs are validated against supplied sources and unvalidated IDs separated."""
    generator = GroundedAnswerGenerator()
    valid_ids = {"[S1]", "[S2]"}

    # Case 1: All valid citations
    text1 = "The Saturn V was used [S1] and landed safely [S2]."
    valid, unvalidated = generator.validate_citations(text1, valid_ids)
    assert valid == ["[S1]", "[S2]"]
    assert unvalidated == []

    # Case 2: Hallucinated / out-of-bounds citations (e.g., model cites [S3] when only [S1], [S2] exist)
    text2 = "Apollo was launched [S1] and reached Mars [S3] using ion drives [S99]."
    valid, unvalidated = generator.validate_citations(text2, valid_ids)
    assert valid == ["[S1]"]
    assert unvalidated == ["[S3]", "[S99]"]

    # Case 3: Deduplication while preserving order
    text3 = "Apollo [S1] was a program [S1] landing on the Moon [S2]."
    valid, unvalidated = generator.validate_citations(text3, valid_ids)
    assert valid == ["[S1]", "[S2]"]
    assert unvalidated == []


def test_missing_api_key_activates_evidence_fallback():
    """Verify that missing API key returns a structured Evidence Viewer response without crashing."""
    generator = GroundedAnswerGenerator(api_key=None, client=None)
    candidates = _create_sample_reranked_results()

    response = generator.generate("What rocket was used?", candidates)

    assert response.is_fallback is True
    assert "Evidence Viewer Mode" in response.answer
    assert "No Gemini API key" in response.fallback_reason
    assert len(response.sources) == 2
    assert response.sources[0].source_id == "[S1]"
    assert response.citations == []
    assert response.model_name is None


def test_successful_generation_with_fake_client():
    """Verify successful generation flow with an injected FakeGenAIClient."""
    fake_client = FakeGenAIClient(
        response_text="The Saturn V rocket was engineered under Wernher von Braun [S1]."
    )
    generator = GroundedAnswerGenerator(
        model_name="gemini-2.5-flash",
        client=fake_client,
    )
    candidates = _create_sample_reranked_results()

    response = generator.generate("Who engineered the rocket?", candidates)

    assert response.is_fallback is False
    assert response.fallback_reason is None
    assert response.model_name == "gemini-2.5-flash"
    assert "Saturn V rocket" in response.answer
    assert response.citations == ["[S1]"]
    assert response.unvalidated_citations == []
    assert len(response.sources) == 2
    assert fake_client.models.last_model == "gemini-2.5-flash"
    assert "User Question: Who engineered the rocket?" in fake_client.models.last_prompt


def test_api_exception_handling_and_redaction():
    """Verify that API runtime errors trigger fallback without leaking sensitive keys."""
    secret_key = "AIzaSyFakeSecretKey12345"
    fake_client = FakeGenAIClient(
        raise_error=RuntimeError(f"HTTP 429: Quota exceeded for key {secret_key}")
    )
    generator = GroundedAnswerGenerator(
        api_key=secret_key,
        client=fake_client,
    )
    candidates = _create_sample_reranked_results()

    response = generator.generate("Question", candidates)

    assert response.is_fallback is True
    assert "Evidence Viewer Mode" in response.answer
    assert "LLM generation failed" in response.answer
    assert secret_key not in response.fallback_reason
    assert "[REDACTED_API_KEY]" in response.fallback_reason
    assert len(response.sources) == 2


def test_generator_empty_query_and_candidates():
    """Verify empty query and empty candidate pool return clean responses without API calls."""
    fake_client = FakeGenAIClient(response_text="Should not be called")
    generator = GroundedAnswerGenerator(client=fake_client)

    # Empty candidates
    res_empty_cands = generator.generate("Valid query", candidates=[])
    assert "No relevant documents or evidence" in res_empty_cands.answer
    assert res_empty_cands.sources == []
    assert fake_client.models.last_prompt is None

    # Empty query
    res_empty_q = generator.generate("", candidates=_create_sample_reranked_results())
    assert "No query was provided" in res_empty_q.answer
    assert res_empty_q.sources == []
    assert fake_client.models.last_prompt is None
