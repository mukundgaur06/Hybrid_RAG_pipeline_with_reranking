"""HybridRAG: Modular Hybrid Retrieval-Augmented Generation Package.

Components:
- Document loaders (PDF, TXT, Markdown)
- Text chunking with rich metadata
- Lexical retrieval (BM25)
- Dense retrieval (Bi-encoder embeddings)
- Hybrid fusion (Reciprocal Rank Fusion - RRF)
- Reranking (Cross-Encoder)
- Generation & Grounding (Gemini API)
"""

__version__ = "0.1.0"
