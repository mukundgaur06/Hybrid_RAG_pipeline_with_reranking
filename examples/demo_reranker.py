"""Demonstration and smoke-test script for Phase 3 Cross-Encoder reranking.

Loads sample documents, chunks them, performs hybrid retrieval (BM25 + Dense)
with Reciprocal Rank Fusion (RRF) to retrieve a candidate pool, and applies
a Cross-Encoder (cross-encoder/ms-marco-MiniLM-L6-v2) to precision-rerank the results.

Usage:
    python examples/demo_reranker.py
    python examples/demo_reranker.py --query "What rocket was used to launch the Apollo missions, and who led its engineering?"
    python examples/demo_reranker.py --query "Which enzyme fixes carbon dioxide during the Calvin cycle?" --candidate-pool 10 --top-k 3
"""

import argparse
from pathlib import Path
from typing import List

from hybrid_rag import (
    BM25Retriever,
    Chunk,
    CrossEncoderReranker,
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
    parser = argparse.ArgumentParser(
        description="HybridRAG Phase 3 Cross-Encoder Reranking Demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What rocket was used to launch the Apollo missions, and who led its engineering?",
        help="Query string to evaluate across the hybrid pipeline and reranker",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=6,
        help="Number of candidates to select from RRF fusion for reranking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Final number of top reranked chunks to display",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HYBRID RAG: PHASE 3 CROSS-ENCODER RERANKING DEMONSTRATION")
    print("=" * 80)

    # 1. Ingest and Chunk
    print("\n[1/5] Ingesting and chunking sample documents...")
    chunks = load_and_chunk_sample_docs()
    print(f"      Total indexed chunks: {len(chunks)}")

    # 2. Build In-Memory Indexes
    print("\n[2/5] Building lexical (BM25) and dense semantic indexes...")
    bm25 = BM25Retriever(chunks=chunks)
    dense = DenseRetriever(chunks=chunks)

    # 3. First-Stage Hybrid Retrieval + RRF Fusion
    query = args.query
    pool_size = max(1, args.candidate_pool)
    top_k = max(1, args.top_k)

    print(f"\n[3/5] First-Stage Hybrid Retrieval & RRF Fusion (Candidate Pool = {pool_size})...")
    print(f'      Query: "{query}"\n')

    # Retrieve candidates from both branches
    bm25_candidates = bm25.retrieve(query, top_k=pool_size)
    dense_candidates = dense.retrieve(query, top_k=pool_size)

    # Blend ranks with RRF (k=60)
    fused_candidates = reciprocal_rank_fusion(
        bm25_candidates, dense_candidates, k=60, top_k=pool_size
    )

    print("-" * 80)
    print(f"STAGE 1 CANDIDATES (Reciprocal Rank Fusion - RRF k=60) [Top {len(fused_candidates)}]")
    print("-" * 80)
    for res in fused_candidates:
        preview = res.chunk.text.replace("\n", " ")[:80]
        page_str = f"Page {res.chunk.page_number}" if res.chunk.page_number else "No Page"
        bm25_info = f"Rank {res.bm25_rank} (score {res.bm25_score:.3f})" if res.bm25_rank else "Not retrieved"
        dense_info = f"Rank {res.dense_rank} (sim {res.dense_score:.3f})" if res.dense_rank else "Not retrieved"
        print(f"RRF Rank {res.rank} | RRF Score: {res.rrf_score:.6f} | ID: {res.chunk.chunk_id}")
        print(f"         Source:     {res.chunk.document_name} ({page_str})")
        print(f"         Provenance: BM25 -> {bm25_info} | Dense -> {dense_info}")
        print(f'         Text:       "{preview}..."\n')

    # 4. Second-Stage Cross-Encoder Precision Reranking
    print("-" * 80)
    print("[4/5] Loading Cross-Encoder: cross-encoder/ms-marco-MiniLM-L6-v2...")
    print("      (Computing full cross-attention over query-passage pairs)")
    reranker = CrossEncoderReranker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L6-v2")

    reranked_results = reranker.rerank(
        query=query, candidates=fused_candidates, top_k=top_k
    )

    # 5. Display Final Reranked Results with Rank Shift Comparison
    print(f"\n[5/5] STAGE 2 RERANKED RESULTS [Top {len(reranked_results)} of {len(fused_candidates)} candidates]")
    print("=" * 80)
    for res in reranked_results:
        preview = res.chunk.text.replace("\n", " ")[:100]
        page_str = f"Page {res.chunk.page_number}" if res.chunk.page_number else "No Page"

        # Calculate rank shift from RRF to Reranker
        if res.rrf_rank is not None:
            shift = res.rrf_rank - res.rerank_rank
            if shift > 0:
                shift_str = f"+{shift} (promoted: RRF Rank {res.rrf_rank} -> Rerank {res.rerank_rank})"
            elif shift < 0:
                shift_str = f"{shift} (demoted: RRF Rank {res.rrf_rank} -> Rerank {res.rerank_rank})"
            else:
                shift_str = f"0 (unchanged at Rank {res.rerank_rank})"
        else:
            shift_str = "N/A"

        print(f"Rerank #{res.rerank_rank} | Cross-Encoder Logit: {res.reranker_score:8.4f} | ID: {res.chunk.chunk_id}")
        print(f"         RRF Pre-Rerank: Rank {res.rrf_rank} (Score: {res.rrf_score:.6f}) | Rank Shift: {shift_str}")
        print(f"         Source:         {res.chunk.document_name} ({page_str})")
        bm25_str = f"Rank {res.bm25_rank} ({res.bm25_score:.3f})" if res.bm25_rank else "Not retrieved"
        dense_str = f"Rank {res.dense_rank} ({res.dense_score:.3f})" if res.dense_rank else "Not retrieved"
        print(f"         Provenance:     BM25: {bm25_str} | Dense: {dense_str}")
        print(f'         Text:           "{preview}..."\n')

    print("=" * 80)
    print("RERANKING OBSERVATIONS:")
    print("1. Cross-Encoder computes joint attention across all query tokens and chunk tokens.")
    print("2. Raw logits are uncalibrated relevance scores (higher = more relevant).")
    print("3. Reranking operates on a small candidate pool to balance precision and inference latency.")
    print("=" * 80)


if __name__ == "__main__":
    main()
