"""Unit and integration tests for HybridRAGPipeline.

Tests document ingestion, index reuse, multi-stage hybrid retrieval with RRF,
Cross-Encoder reranking, and grounded generation flow using injected test doubles.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pytest

from hybrid_rag.generator import GroundedAnswerGenerator
from hybrid_rag.pipeline import HybridRAGPipeline
from hybrid_rag.reranker import CrossEncoderReranker
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.dense import DenseRetriever


class TrackingEmbeddingModel:
    """Fake embedding model tracking how many times encode() is called."""

    def __init__(self, vector_dim: int = 3):
        self.vector_dim = vector_dim
        self.call_count = 0
        self.encoded_text_batches: List[List[str]] = []

    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.call_count += 1
        self.encoded_text_batches.append(texts)
        # Generate predictable deterministic pseudo-vectors
        vectors = []
        for text in texts:
            # Simple hash-based vector for reproducibility
            val = float(len(text) % 10) / 10.0
            vectors.append([val, 1.0 - val, 0.5])
        return np.array(vectors, dtype=np.float32)


class FakeCrossEncoder:
    """Fake Cross-Encoder returning deterministic relevance scores."""

    def __init__(self, default_score: float = 2.5):
        self.default_score = default_score
        self.predict_calls: List[List[Tuple[str, str]]] = []

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        self.predict_calls.append(pairs)
        # Score higher if query word is in passage
        scores = []
        for q, p in pairs:
            bonus = 2.0 if any(w.lower() in p.lower() for w in q.split()) else 0.0
            scores.append(self.default_score + bonus)
        return np.array(scores, dtype=np.float32)


class FakeGenAIResponse:
    def __init__(self, text: str):
        self.text = text


class FakeGenAIClient:
    def __init__(self, response_text: str = "Ground answer [S1]."):
        self.response_text = response_text
        self.models = self

    def generate_content(self, model: str, contents: str, **kwargs) -> FakeGenAIResponse:
        return FakeGenAIResponse(text=self.response_text)


@pytest.fixture
def sample_doc_path(tmp_path: Path) -> Path:
    """Create a temporary text file for testing pipeline ingestion."""
    doc_file = tmp_path / "space_test.txt"
    doc_file.write_text(
        "The Saturn V rocket launched the Apollo 11 crew to the Moon in July 1969.\n"
        "Neil Armstrong and Buzz Aldrin were the first humans to walk on the lunar surface.\n"
        "Michael Collins remained in orbit aboard the Command Module Columbia.",
        encoding="utf-8",
    )
    return doc_file


def test_pipeline_ingestion_and_index_reuse(sample_doc_path: Path):
    """Verify that chunks are indexed once and repeated queries do NOT re-embed documents."""
    fake_embedding = TrackingEmbeddingModel()
    fake_reranker_model = FakeCrossEncoder()
    fake_client = FakeGenAIClient(response_text="The Saturn V was used [S1].")

    dense = DenseRetriever(model=fake_embedding)
    reranker = CrossEncoderReranker(model=fake_reranker_model)
    generator = GroundedAnswerGenerator(client=fake_client)

    pipeline = HybridRAGPipeline(
        dense_retriever=dense,
        reranker=reranker,
        generator=generator,
    )

    # 1. Ingest document
    total_chunks = pipeline.ingest_documents([sample_doc_path])
    assert total_chunks > 0
    assert pipeline.total_chunks == total_chunks

    # Embedding model was called once during document indexing
    assert fake_embedding.call_count == 1
    chunk_batch_size = len(fake_embedding.encoded_text_batches[0])
    assert chunk_batch_size == total_chunks

    # 2. First query
    res1 = pipeline.query("Which rocket launched Apollo 11?")
    assert res1.is_fallback is False
    assert res1.citations == ["[S1]"]
    # Embedding model called once more, ONLY for the single query text
    assert fake_embedding.call_count == 2
    assert len(fake_embedding.encoded_text_batches[1]) == 1

    # 3. Second query
    res2 = pipeline.query("Who walked on the Moon?")
    assert res2.is_fallback is False
    # Embedding model called once more, ONLY for the second query text
    # Crucially, the document chunks were NOT re-embedded!
    assert fake_embedding.call_count == 3
    assert len(fake_embedding.encoded_text_batches[2]) == 1


def test_pipeline_provenance_preservation(sample_doc_path: Path):
    """Verify that retrieval scores, ranks, and metadata are preserved through to response sources."""
    fake_embedding = TrackingEmbeddingModel()
    fake_reranker_model = FakeCrossEncoder()
    fake_client = FakeGenAIClient(response_text="Answer with citation [S1].")

    pipeline = HybridRAGPipeline(
        dense_retriever=DenseRetriever(model=fake_embedding),
        reranker=CrossEncoderReranker(model=fake_reranker_model),
        generator=GroundedAnswerGenerator(client=fake_client),
        candidate_pool_size=4,
        final_context_size=2,
    )

    pipeline.ingest_documents([sample_doc_path])
    response = pipeline.query("Saturn V Apollo")

    assert len(response.sources) <= 2
    for src in response.sources:
        # Check chunk attributes
        assert src.chunk.document_name == "space_test.txt"
        assert src.chunk.chunk_id is not None
        assert src.chunk.text != ""
        # Check RerankedResult provenance attributes
        assert src.reranked_result is not None
        assert src.reranked_result.rerank_rank >= 1
        assert src.reranked_result.reranker_score is not None
        assert src.reranked_result.rrf_score is not None


def test_pipeline_empty_corpus_and_invalid_queries():
    """Verify clean handling of empty index, empty query, and whitespace queries."""
    pipeline = HybridRAGPipeline()

    # Querying empty pipeline
    res_empty = pipeline.query("Test query without ingesting")
    assert "document index is empty" in res_empty.answer
    assert res_empty.sources == []

    # Empty query string
    res_no_q = pipeline.query("")
    assert "No query was provided" in res_no_q.answer
    assert res_no_q.sources == []

    # Whitespace query
    res_ws = pipeline.query("   \t\n  ")
    assert "No query was provided" in res_ws.answer
    assert res_ws.sources == []


def test_pipeline_evidence_fallback_when_no_api_key(sample_doc_path: Path):
    """Verify that with no API key and no client, pipeline returns structured evidence fallback."""
    fake_embedding = TrackingEmbeddingModel()
    fake_reranker_model = FakeCrossEncoder()

    # Generator without client or API key
    generator = GroundedAnswerGenerator(api_key=None, client=None)

    pipeline = HybridRAGPipeline(
        dense_retriever=DenseRetriever(model=fake_embedding),
        reranker=CrossEncoderReranker(model=fake_reranker_model),
        generator=generator,
    )

    pipeline.ingest_documents([sample_doc_path])
    response = pipeline.query("What rocket was used?")

    assert response.is_fallback is True
    assert "Evidence Viewer Mode" in response.answer
    assert len(response.sources) > 0
    assert response.citations == []
