"""Streamlit Web Application for HybridRAG.

Provides an interactive user interface for uploading documents (PDF, TXT, MD),
indexing chunks in memory, executing multi-stage hybrid retrieval with RRF
and Cross-Encoder reranking, and inspecting grounded answers with verified citations.
"""

import os
from pathlib import Path
import tempfile
from typing import List, Optional

import streamlit as st

from hybrid_rag import (
    CrossEncoderReranker,
    DenseRetriever,
    GroundedAnswerGenerator,
    HybridRAGPipeline,
    PipelineResponse,
)


@st.cache_resource(show_spinner="Loading dense embedding model...")
def get_dense_embedding_model():
    """Cache the dense SentenceTransformer model in memory across sessions."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading Cross-Encoder reranker model...")
def get_cross_encoder_model():
    """Cache the CrossEncoder reranker model in memory across sessions."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def initialize_session_pipeline(api_key: Optional[str] = None) -> HybridRAGPipeline:
    """Initialize a session-isolated pipeline with shared cached models."""
    dense = DenseRetriever(model=get_dense_embedding_model())
    reranker = CrossEncoderReranker(model=get_cross_encoder_model())
    generator = GroundedAnswerGenerator(api_key=api_key)
    return HybridRAGPipeline(
        dense_retriever=dense,
        reranker=reranker,
        generator=generator,
        candidate_pool_size=6,
        final_context_size=3,
    )


def inject_custom_css():
    """Inject custom styles for a modern, high-contrast, polished interface."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            color: #94a3b8;
            font-size: 1.0rem;
            margin-bottom: 1.5rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            margin: 0.2rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: #1e293b;
            color: #38bdf8;
            border: 1px solid #334155;
        }
        .badge-active {
            background-color: #064e3b;
            color: #34d399;
            border: 1px solid #059669;
        }
        .badge-fallback {
            background-color: #78350f;
            color: #fbbf24;
            border: 1px solid #b45309;
        }
        .source-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .citation-tag {
            color: #38bdf8;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="HybridRAG Pipeline",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Sidebar: Setup & Document Management
    with st.sidebar:
        st.markdown("### ⚡ HybridRAG Controls")

        # API Key Configuration
        existing_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        user_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            value=existing_key or "",
            help="If not provided, the pipeline runs safely in Evidence Viewer mode without calling the LLM.",
        )
        api_key = user_key.strip() if user_key else existing_key

        # Initialize or update session pipeline
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = initialize_session_pipeline(api_key=api_key)
            st.session_state.indexed_files = []
            st.session_state.total_chunks = 0
            st.session_state.last_response = None
        else:
            # Update generator API key if user changed input
            st.session_state.pipeline.generator._api_key = api_key
            st.session_state.pipeline.generator._client = None

        # Status Badge
        has_key = bool(api_key)
        if has_key:
            st.markdown('<span class="badge badge-active">● Gemini Synthesis: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-fallback">▲ Evidence Viewer (No API Key)</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📂 Document Ingestion")

        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Uploaded documents are processed in memory and never stored in the Git repository.",
        )

        col_ingest, col_sample = st.columns(2)

        with col_ingest:
            if st.button("📥 Index Files", use_container_width=True):
                if uploaded_files:
                    with st.spinner("Ingesting and indexing chunks..."):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            temp_paths: List[Path] = []
                            for uploaded in uploaded_files:
                                temp_file = Path(tmp_dir) / uploaded.name
                                temp_file.write_bytes(uploaded.getvalue())
                                temp_paths.append(temp_file)

                            count = st.session_state.pipeline.ingest_documents(temp_paths)
                            st.session_state.total_chunks = count
                            st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"Indexed {count} chunks!")
                else:
                    st.warning("Select files to upload first.")

        with col_sample:
            if st.button("🚀 Load Samples", use_container_width=True):
                sample_dir = Path(__file__).resolve().parent.parent.parent.parent / "sample_docs"
                sample_files = sorted(
                    [f for f in sample_dir.glob("*.*") if f.suffix.lower() in {".md", ".txt", ".pdf"}]
                )
                if sample_files:
                    with st.spinner("Indexing sample documents..."):
                        count = st.session_state.pipeline.ingest_documents(sample_files)
                        st.session_state.total_chunks = count
                        st.session_state.indexed_files = [f.name for f in sample_files]
                    st.success(f"Indexed {count} sample chunks!")
                else:
                    st.error("Sample directory not found.")

        # Index Status
        st.markdown("---")
        st.markdown("#### 📊 Session Index Stats")
        st.metric("Total Chunks", st.session_state.total_chunks)
        if st.session_state.indexed_files:
            st.caption(f"Files: {', '.join(st.session_state.indexed_files)}")
        else:
            st.caption("No documents currently indexed.")

        # Clear Index Action
        if st.button("🗑️ Clear Index & Reset", use_container_width=True):
            st.session_state.pipeline = initialize_session_pipeline(api_key=api_key)
            st.session_state.indexed_files = []
            st.session_state.total_chunks = 0
            st.session_state.last_response = None
            st.rerun()

    # Main Panel
    st.markdown('<div class="main-header">⚡ HybridRAG: Multi-Stage Document Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Lexical Search (BM25) • Dense Semantic Search (MiniLM) • '
        'Reciprocal Rank Fusion (RRF) • Cross-Encoder Reranking • Grounded Gemini Synthesis</div>',
        unsafe_allow_html=True,
    )

    # Empty State Notice
    if st.session_state.total_chunks == 0:
        st.info("👋 Welcome! Use the sidebar to upload files or click **🚀 Load Samples** to explore the pipeline.")

    # Query Input Box
    query_text = st.text_input(
        "Ask a question about the indexed documents:",
        placeholder="e.g. What rocket was used to launch the Apollo missions, and who led its engineering?",
    )

    # Quick prompt buttons for convenience
    if st.session_state.total_chunks > 0:
        col_q1, col_q2, col_q3 = st.columns(3)
        if col_q1.button("Apollo Saturn V"):
            query_text = "What rocket was used to launch the Apollo missions, and who led its engineering?"
        if col_q2.button("Calvin Cycle Enzyme"):
            query_text = "Which enzyme fixes atmospheric carbon dioxide in the Calvin cycle?"
        if col_q3.button("Four Nines SLA"):
            query_text = "How many minutes of downtime are permitted per year under a four nines SLA?"

    if st.button("🔍 Run Pipeline", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a question.")
        elif st.session_state.total_chunks == 0:
            st.error("No documents indexed. Ingest documents in the sidebar first.")
        else:
            with st.spinner("Executing hybrid retrieval, fusion, reranking, and generation..."):
                response = st.session_state.pipeline.query(query_text)
                st.session_state.last_response = response

    # Display Response
    if st.session_state.last_response:
        res: PipelineResponse = st.session_state.last_response

        st.markdown("---")

        # Mode Banner
        if res.is_fallback:
            st.warning(f"⚠️ **Evidence Viewer Mode**: {res.fallback_reason}")
        else:
            st.success(f"✨ **Grounded Gemini Synthesis** ({res.model_name})")

        tab_answer, tab_evidence, tab_provenance = st.tabs(
            ["📝 Grounded Answer", "🔍 Reranked Evidence", "🔬 Pipeline Provenance"]
        )

        with tab_answer:
            st.markdown("### Answer")
            st.markdown(res.answer)

            if res.citations:
                st.markdown(f"**Verified Source Citations**: `{'`, `'.join(res.citations)}`")
            if res.unvalidated_citations:
                st.markdown(
                    f"⚠️ **Unvalidated Citations Detected**: `{'`, `'.join(res.unvalidated_citations)}` "
                    "(referenced by LLM but not present in supplied context)"
                )

        with tab_evidence:
            st.markdown(f"### Evidence Passages Passed to Generator ({len(res.sources)})")
            for src in res.sources:
                page_str = f"Page {src.chunk.page_number}" if src.chunk.page_number else "No Page"
                score_str = (
                    f"Reranker Logit: {src.reranked_result.reranker_score:.4f}"
                    if src.reranked_result
                    else "N/A"
                )
                with st.expander(
                    f"Source {src.source_id}: {src.chunk.document_name} ({page_str}) — {score_str}",
                    expanded=True,
                ):
                    st.write(src.chunk.text)
                    if src.reranked_result:
                        rr = src.reranked_result
                        st.caption(
                            f"Post-Rerank Rank: #{rr.rerank_rank} | Pre-Rerank RRF Rank: #{rr.rrf_rank} "
                            f"(RRF Score: {rr.rrf_score:.5f})"
                        )

        with tab_provenance:
            st.markdown("### Multi-Stage Retrieval & Fusion Provenance")
            st.markdown(
                "Compare candidate ranks and scores as they progress from Stage 1 (BM25 + Dense) "
                "through RRF fusion to Stage 2 Cross-Encoder reranking:"
            )

            prov_rows = []
            for src in res.sources:
                if src.reranked_result:
                    rr = src.reranked_result
                    prov_rows.append(
                        {
                            "Source ID": src.source_id,
                            "Chunk ID": src.chunk.chunk_id,
                            "File": src.chunk.document_name,
                            "Rerank Rank": rr.rerank_rank,
                            "Reranker Logit": round(rr.reranker_score, 4),
                            "RRF Rank": rr.rrf_rank,
                            "RRF Score": round(rr.rrf_score, 6) if rr.rrf_score else None,
                            "BM25 Rank": rr.bm25_rank,
                            "Dense Rank": rr.dense_rank,
                        }
                    )
            if prov_rows:
                st.dataframe(prov_rows, use_container_width=True)


if __name__ == "__main__":
    main()
