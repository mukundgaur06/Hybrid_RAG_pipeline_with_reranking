"""Demonstration and smoke-test script for Phase 2 retrieval.

Loads sample documents, chunks them, builds both BM25 and Dense in-memory indexes,
and demonstrates Lexical, Dense, and Reciprocal Rank Fusion (RRF) retrieval.

Usage:
    python examples/demo_retrieval.py
    python examples/demo_retrieval.py --query "Which enzyme fixes carbon dioxide in the Calvin cycle?"
"""

import argparse
from pathlib import Path
from typing import List

from hybrid_rag import (
    BM25Retriever,
    Chunk,
    DenseRetriever,
    DocumentLoader,
    TextChunker,
    reciprocal_rank_fusion,
)


def load_and_chunk_sample_docs() -> List[Chunk]:
    """Ingest all sample documents and partition them into overlapping chunks."""
    sample_dir = Path(__file__).resolve().parent.parent / "sample_docs"
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)

    all_chunks: List[Chunk] = []
    sample_files = sorted(sample_dir.glob("*.*"))

    for f in sample_files:
        if f.suffix.lower() in {".md", ".txt", ".pdf"}:
            doc = loader.load(f)
            doc_chunks = chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="HybridRAG Phase 2 Retrieval Demo")
    parser.add_argument(
        "--query",
        type=str,
        default="What rocket was used to launch the Apollo missions, and who led its engineering?",
        help="Query string to evaluate across retrievers",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidates to display per retriever",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HYBRID RAG: PHASE 2 RETRIEVAL DEMONSTRATION")
    print("=" * 80)

    # 1. Ingest and Chunk
    print("\n[1/4] Ingesting and chunking sample documents...")
    chunks = load_and_chunk_sample_docs()
    print(f"      Total indexed chunks: {len(chunks)}")

    # 2. Build BM25 Index
    print("\n[2/4] Building in-memory BM25 lexical index...")
    bm25 = BM25Retriever(chunks=chunks)

    # 3. Build Dense Index
    print("\n[3/4] Building in-memory Dense vector index (all-MiniLM-L6-v2)...")
    dense = DenseRetriever(chunks=chunks)

    # 4. Execute Retrieval
    query = args.query
    top_k = args.top_k
    print(f"\n[4/4] Executing query across retrievers:")
    print(f'      Query: "{query}"\n')

    bm25_results = bm25.retrieve(query, top_k=top_k)
    dense_results = dense.retrieve(query, top_k=top_k)
    fused_results = reciprocal_rank_fusion(bm25_results, dense_results, k=60, top_k=top_k)

    # Display BM25
    print("-" * 80)
    print(f"A. LEXICAL RETRIEVAL (BM25) [Top {len(bm25_results)}]")
    print("-" * 80)
    for res in bm25_results:
        preview = res.chunk.text.replace("\n", " ")[:90]
        page_str = f"Page {res.chunk.page_number}" if res.chunk.page_number else "No Page"
        print(f"Rank {res.rank} | Score: {res.score:8.4f} | ID: {res.chunk.chunk_id}")
        print(f"       Source: {res.chunk.document_name} ({page_str})")
        print(f'       Text:   "{preview}..."\n')

    # Display Dense
    print("-" * 80)
    print(f"B. DENSE SEMANTIC RETRIEVAL (Cosine Similarity) [Top {len(dense_results)}]")
    print("-" * 80)
    for res in dense_results:
        preview = res.chunk.text.replace("\n", " ")[:90]
        page_str = f"Page {res.chunk.page_number}" if res.chunk.page_number else "No Page"
        print(f"Rank {res.rank} | Cosine Sim: {res.score:6.4f} | ID: {res.chunk.chunk_id}")
        print(f"       Source: {res.chunk.document_name} ({page_str})")
        print(f'       Text:   "{preview}..."\n')

    # Display Fused (RRF)
    print("-" * 80)
    print(f"C. HYBRID RETRIEVAL (Reciprocal Rank Fusion - RRF k=60) [Top {len(fused_results)}]")
    print("-" * 80)
    for res in fused_results:
        preview = res.chunk.text.replace("\n", " ")[:90]
        page_str = f"Page {res.chunk.page_number}" if res.chunk.page_number else "No Page"
        bm25_info = f"Rank {res.bm25_rank} (score {res.bm25_score:.3f})" if res.bm25_rank else "Not retrieved"
        dense_info = f"Rank {res.dense_rank} (sim {res.dense_score:.3f})" if res.dense_rank else "Not retrieved"
        print(f"Rank {res.rank} | RRF Score: {res.rrf_score:.6f} | ID: {res.chunk.chunk_id}")
        print(f"       Source:     {res.chunk.document_name} ({page_str})")
        print(f"       Provenance: BM25 -> {bm25_info} | Dense -> {dense_info}")
        print(f'       Text:       "{preview}..."\n')


if __name__ == "__main__":
    main()
