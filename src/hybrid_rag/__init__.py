"""HybridRAG: Modular Hybrid Retrieval-Augmented Generation Package.

Components:
- Document loaders (PDF, TXT, Markdown)
- Text chunking with rich metadata preservation
- Lexical retrieval (BM25)
- Dense retrieval (Bi-encoder embeddings)
- Hybrid fusion (Reciprocal Rank Fusion - RRF)
- Reranking (Cross-Encoder)
- Generation & Grounding (Gemini API)
"""

from hybrid_rag.chunker import TextChunker
from hybrid_rag.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    HybridRAGError,
    UnsupportedFileTypeError,
)
from hybrid_rag.generator import GroundedAnswerGenerator
from hybrid_rag.loader import DocumentLoader
from hybrid_rag.pipeline import HybridRAGPipeline
from hybrid_rag.reranker import CrossEncoderReranker
from hybrid_rag.retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion, tokenize
from hybrid_rag.schema import (
    Chunk,
    Document,
    DocumentPage,
    FusedResult,
    PipelineResponse,
    RerankedResult,
    RetrievalResult,
    SourcePassage,
)
from hybrid_rag.evaluation import (
    calculate_mrr,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)

__version__ = "0.1.0"

__all__ = [
    "DocumentLoader",
    "TextChunker",
    "Document",
    "DocumentPage",
    "Chunk",
    "RetrievalResult",
    "FusedResult",
    "RerankedResult",
    "SourcePassage",
    "PipelineResponse",
    "BM25Retriever",
    "tokenize",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "GroundedAnswerGenerator",
    "HybridRAGPipeline",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
    "calculate_mrr",
    "HybridRAGError",
    "UnsupportedFileTypeError",
    "EmptyDocumentError",
    "DocumentParsingError",
]

