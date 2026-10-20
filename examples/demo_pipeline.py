"""Demonstration script for Phase 4: Integrated HybridRAG Pipeline.

Demonstrates end-to-end document ingestion, hybrid retrieval (BM25 + Dense),
Reciprocal Rank Fusion, Cross-Encoder reranking, and strictly grounded
answer generation using Gemini (or safe Evidence Viewer fallback if no API key is set).

Usage:
    # Run in Evidence Viewer mode (default if no API key is configured)
    python examples/demo_pipeline.py

    # Query with custom question
    python examples/demo_pipeline.py --query "What rocket launched Apollo 11, and who engineered it?"

    # Run with Gemini API key (safely passed via environment variable)
    $env:GEMINI_API_KEY="your-api-key"
    python examples/demo_pipeline.py
"""

import argparse
import os
from pathlib import Path

from hybrid_rag import HybridRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="HybridRAG Phase 4 End-to-End Pipeline Demo")
    parser.add_argument(
        "--query",
        type=str,
        default="What rocket was used to launch the Apollo missions, and who led its engineering?",
        help="Query string to evaluate through the complete pipeline",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=6,
        help="Number of candidates to retrieve from Stage 1 (BM25 + Dense) before fusion and reranking",
    )
    parser.add_argument(
        "--final-context",
        type=int,
        default=3,
        help="Number of top reranked chunks to feed as evidence to the generator",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HYBRID RAG: PHASE 4 INTEGRATED PIPELINE DEMONSTRATION")
    print("=" * 80)

    # 1. Locate sample documents
    sample_dir = Path(__file__).resolve().parent.parent / "sample_docs"
    sample_files = sorted(
        [f for f in sample_dir.glob("*.*") if f.suffix.lower() in {".md", ".txt", ".pdf"}]
    )

    print(f"\n[1/3] Ingesting {len(sample_files)} sample documents...")
    for f in sample_files:
        print(f"      - {f.name}")

    # 2. Initialize pipeline
    # Note: GroundedAnswerGenerator automatically checks GEMINI_API_KEY / GOOGLE_API_KEY
    pipeline = HybridRAGPipeline(
        candidate_pool_size=args.candidate_pool,
        final_context_size=args.final_context,
    )
    indexed_chunks = pipeline.ingest_documents(sample_files)
    print(f"      Successfully indexed {indexed_chunks} chunks into in-memory BM25 & Dense indexes.")

    # 3. Query the pipeline
    query = args.query
    print(f"\n[2/3] Executing Query:")
    print(f'      "{query}"\n')

    response = pipeline.query(query)

    # 4. Display Results
    print("=" * 80)
    print("PIPELINE RESPONSE & EVIDENCE REPORT")
    print("=" * 80)

    print(f"\nMode: {'[FALLBACK: Evidence Viewer]' if response.is_fallback else '[ACTIVE: Gemini Grounded Synthesis]'}")
    if response.is_fallback:
        print(f"Fallback Notice: {response.fallback_reason}")
    else:
        print(f"Generation Model: {response.model_name}")

    print("\n" + "-" * 80)
    print("ANSWER:")
    print("-" * 80)
    print(response.answer)

    if response.citations:
        print(f"\nValidated Citations: {', '.join(response.citations)}")
    if response.unvalidated_citations:
        print(f"Unvalidated / Extraneous Citations: {', '.join(response.unvalidated_citations)}")

    print("\n" + "-" * 80)
    print(f"RETRIEVED & RERANKED EVIDENCE PASSAGES ({len(response.sources)} passed to generator):")
    print("-" * 80)

    for src in response.sources:
        chunk = src.chunk
        page_str = f"Page {chunk.page_number}" if chunk.page_number else "No Page"
        rerank_info = ""
        if src.reranked_result:
            rr = src.reranked_result
            rerank_info = (
                f" | Rerank #{rr.rerank_rank} (Logit: {rr.reranker_score:.3f}) "
                f"| Pre-Rerank RRF Rank: {rr.rrf_rank}"
            )

        print(f"Source {src.source_id} [{chunk.document_name} ({page_str})]{rerank_info}")
        preview = chunk.text.replace("\n", " ").strip()
        print(f'Text: "{preview[:140]}..."\n')

    print("=" * 80)
    print("PIPELINE ARCHITECTURAL NOTES:")
    print("1. Indexes are built once; repeated queries do not re-encode documents.")
    print("2. The generator receives ONLY the top reranked passages, tagged as [S1], [S2]...")
    print("3. When no API key is set, the pipeline operates safely in Evidence Viewer mode.")
    print("4. Citations in the generated answer are strictly validated against supplied source IDs.")
    print("=" * 80)


if __name__ == "__main__":
    main()
