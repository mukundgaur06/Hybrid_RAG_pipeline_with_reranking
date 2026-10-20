"""Unit tests for BM25Retriever.

Verifies consistent tokenization, lexical ranking, 1-based ranks,
empty query handling, empty index handling, and metadata preservation.
"""

import pytest

from hybrid_rag.retrieval.bm25 import BM25Retriever, tokenize
from hybrid_rag.schema import Chunk


def test_tokenize_consistency():
    """Verify tokenizer lowercases, strips punctuation, and handles empty inputs."""
    text = "The Saturn V rocket (Apollo 11) generated 34.5 meganewtons!"
    tokens = tokenize(text)
    assert tokens == ["the", "saturn", "v", "rocket", "apollo", "11", "generated", "34", "5", "meganewtons"]

    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize("!@#$%^&*()") == []


def test_bm25_ranking_and_metadata():
    """Verify that BM25 ranks matching chunks highest and preserves chunk provenance."""
    c1 = Chunk(
        chunk_id="chunk_space_01",
        document_id="doc_space",
        document_name="space.md",
        page_number=1,
        chunk_index=0,
        text="Saturn V super heavy-lift rocket engineered by Wernher von Braun.",
        char_start=0,
        char_end=65,
    )
    c2 = Chunk(
        chunk_id="chunk_bio_02",
        document_id="doc_bio",
        document_name="bio.txt",
        page_number=1,
        chunk_index=0,
        text="Photosynthesis in chloroplasts converts light into glucose and oxygen.",
        char_start=0,
        char_end=70,
    )
    c3 = Chunk(
        chunk_id="chunk_cloud_03",
        document_id="doc_cloud",
        document_name="cloud.md",
        page_number=None,
        chunk_index=0,
        text="High availability cloud architecture with 99.99 percent uptime SLA.",
        char_start=0,
        char_end=67,
    )

    retriever = BM25Retriever(chunks=[c1, c2, c3])
    assert retriever.total_chunks == 3

    # Query targeting space
    results = retriever.retrieve(query="Saturn V rocket", top_k=2)

    assert len(results) == 1  # Only c1 has matching query tokens
    top = results[0]
    assert top.chunk.chunk_id == "chunk_space_01"
    assert top.rank == 1
    assert top.score > 0.0
    assert top.chunk.document_name == "space.md"
    assert top.chunk.page_number == 1


def test_bm25_empty_query_and_empty_index():
    """Verify that empty queries or empty indexes return empty result lists."""
    c1 = Chunk(
        chunk_id="c1",
        document_id="d1",
        document_name="d1.txt",
        page_number=None,
        chunk_index=0,
        text="Sample text content for retrieval.",
        char_start=0,
        char_end=34,
    )

    retriever = BM25Retriever([c1])

    # Empty query
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

    # Non-alphanumeric query that yields zero tokens
    assert retriever.retrieve("??? ### !!!") == []

    # Empty index
    empty_retriever = BM25Retriever()
    assert empty_retriever.retrieve("sample") == []


def test_bm25_top_k_bounds_and_filtering():
    """Verify top_k bounds and filter_unmatched behavior."""
    chunks = [
        Chunk(f"c_{i}", "d1", "d1.txt", None, i, f"Token{i} and common keyword", 0, 25)
        for i in range(10)
    ]
    retriever = BM25Retriever(chunks)

    # top_k=3 should return at most 3 items with 1-based ranks 1, 2, 3
    results = retriever.retrieve("common keyword", top_k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]

    # top_k <= 0 should return empty list
    assert retriever.retrieve("common keyword", top_k=0) == []
    assert retriever.retrieve("common keyword", top_k=-5) == []

    # Query with no match anywhere
    assert retriever.retrieve("nonexistentterm12345", filter_unmatched=True) == []


def test_bm25_deterministic_tie_breaking():
    """Verify that identical BM25 scores are deterministically sorted by chunk_id."""
    c_b = Chunk("chunk_b", "d1", "d1.txt", None, 0, "alpha beta", 0, 10)
    c_a = Chunk("chunk_a", "d1", "d1.txt", None, 1, "alpha beta", 0, 10)

    retriever = BM25Retriever([c_b, c_a])
    results = retriever.retrieve("alpha", top_k=2)

    assert len(results) == 2
    assert results[0].score == results[1].score
    # Should be sorted alphabetically by chunk_id
    assert results[0].chunk.chunk_id == "chunk_a"
    assert results[1].chunk.chunk_id == "chunk_b"
