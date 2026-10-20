"""Unit tests for DenseRetriever.

Uses an injected test double (FakeEmbeddingModel) to provide 100% offline,
deterministic testing of cosine similarity calculations, top-k ranking,
and failure mode handling without network access or model downloads.
"""

from typing import Dict, List
import numpy as np
import pytest

from hybrid_rag.retrieval.dense import DenseRetriever
from hybrid_rag.schema import Chunk


class FakeEmbeddingModel:
    """Test double returning controlled, deterministic vectors for offline tests."""

    def __init__(self, vector_map: Dict[str, List[float]], default_dim: int = 3):
        self.vector_map = vector_map
        self.default_dim = default_dim

    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        vectors = []
        for text in texts:
            if text in self.vector_map:
                vectors.append(self.vector_map[text])
            else:
                vectors.append([0.0] * self.default_dim)
        return np.array(vectors, dtype=np.float32)


def test_dense_similarity_calculation_and_ranking():
    """Verify exact cosine similarity with known orthogonal, parallel, and opposite vectors."""
    # Define 3D unit vectors
    fake_model = FakeEmbeddingModel(
        vector_map={
            "query_text": [1.0, 0.0, 0.0],
            "chunk_aligned": [2.0, 0.0, 0.0],    # Parallel (cosine = 1.0)
            "chunk_orthogonal": [0.0, 5.0, 0.0], # Orthogonal (cosine = 0.0)
            "chunk_opposite": [-3.0, 0.0, 0.0],  # Opposite (cosine = -1.0)
        }
    )

    c1 = Chunk("c_aligned", "doc1", "doc1.txt", 1, 0, "chunk_aligned", 0, 13)
    c2 = Chunk("c_orthogonal", "doc1", "doc1.txt", 1, 1, "chunk_orthogonal", 14, 30)
    c3 = Chunk("c_opposite", "doc1", "doc1.txt", 1, 2, "chunk_opposite", 31, 45)

    retriever = DenseRetriever(model=fake_model, chunks=[c1, c2, c3])
    assert retriever.total_chunks == 3

    results = retriever.retrieve("query_text", top_k=3)

    assert len(results) == 3

    # Rank 1: Aligned chunk, similarity = 1.0
    assert results[0].chunk.chunk_id == "c_aligned"
    assert results[0].rank == 1
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0
    assert results[0].chunk.document_name == "doc1.txt"

    # Rank 2: Orthogonal chunk, similarity = 0.0
    assert results[1].chunk.chunk_id == "c_orthogonal"
    assert results[1].rank == 2
    assert pytest.approx(results[1].score, abs=1e-5) == 0.0

    # Rank 3: Opposite chunk, similarity = -1.0
    assert results[2].chunk.chunk_id == "c_opposite"
    assert results[2].rank == 3
    assert pytest.approx(results[2].score, abs=1e-5) == -1.0


def test_dense_empty_inputs_and_bounds():
    """Verify that empty queries, empty collections, and non-positive top_k return empty lists."""
    fake_model = FakeEmbeddingModel({"text": [1.0, 0.0, 0.0]})
    chunk = Chunk("c1", "d1", "d.txt", None, 0, "text", 0, 4)

    # Empty collection
    empty_retriever = DenseRetriever(model=fake_model, chunks=[])
    assert empty_retriever.retrieve("query") == []

    # Valid collection with empty queries
    retriever = DenseRetriever(model=fake_model, chunks=[chunk])
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \n\t") == []

    # top_k bounds
    assert retriever.retrieve("text", top_k=0) == []
    assert retriever.retrieve("text", top_k=-10) == []

    # top_k exceeding available chunks returns all available chunks
    results = retriever.retrieve("text", top_k=50)
    assert len(results) == 1
    assert results[0].rank == 1


def test_dense_non_finite_embeddings_raises():
    """Verify that non-finite vectors (NaN, Inf) are caught and raise a ValueError."""
    fake_model = FakeEmbeddingModel({"nan_chunk": [float("nan"), 0.0, 1.0]})
    bad_chunk = Chunk("bad", "d1", "d.txt", None, 0, "nan_chunk", 0, 9)

    retriever = DenseRetriever(model=fake_model)
    with pytest.raises(ValueError, match="non-finite values"):
        retriever.index_chunks([bad_chunk])


def test_dense_deterministic_tie_breaking():
    """Verify that identical cosine similarity scores are deterministically sorted by chunk_id."""
    fake_model = FakeEmbeddingModel(
        vector_map={
            "query": [1.0, 0.0, 0.0],
            "text_b": [1.0, 0.0, 0.0],
            "text_a": [1.0, 0.0, 0.0],
        }
    )
    cb = Chunk("chunk_b", "d1", "d.txt", None, 0, "text_b", 0, 6)
    ca = Chunk("chunk_a", "d1", "d.txt", None, 1, "text_a", 0, 6)

    retriever = DenseRetriever(model=fake_model, chunks=[cb, ca])
    results = retriever.retrieve("query", top_k=2)

    assert len(results) == 2
    assert results[0].score == results[1].score
    # Should sort alphabetically by chunk_id
    assert results[0].chunk.chunk_id == "chunk_a"
    assert results[1].chunk.chunk_id == "chunk_b"
