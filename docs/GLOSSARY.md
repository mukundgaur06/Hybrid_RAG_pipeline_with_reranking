# HybridRAG: Engineering & ML Glossary

This glossary provides crystal-clear, interview-ready explanations of the core mathematical and software concepts used in HybridRAG.

---

## 1. Chunking
- **Definition**: The process of breaking down a large unstructured document into smaller, coherent text segments (chunks) before indexing.
- **Why it matters**: LLMs and embedding models have finite context windows (e.g., 256 or 512 tokens for bi-encoders). If you pass an entire 50-page PDF into an embedding model, the vector representation gets diluted, and you lose fine-grained semantic resolution.
- **Chunk Size & Overlap**:
  - *Chunk Size*: Number of characters or tokens per segment (e.g., 500 characters).
  - *Chunk Overlap*: A shared margin of text between consecutive chunks (e.g., 100 characters). Overlap helps preserve local context and continuity across boundaries, reducing semantic fragmentation. However, fixed character overlap does not guarantee that every complete sentence will appear intact within a single chunk.

---

## 2. Embeddings
- **Definition**: Dense numerical vector representations of text in a continuous, high-dimensional vector space (e.g., 384 dimensions for `all-MiniLM-L6-v2`).
- **How it works**: Deep neural networks (Transformers) are trained using contrastive learning so that semantically similar sentences are mapped to vectors that are geometrically close to one another in vector space, regardless of the specific words used.
- **Interview Takeaway**: Unlike sparse one-hot encodings where vectors are orthogonal and proportional to vocabulary size, dense embeddings capture semantic meaning, synonyms, and analogies.

---

## 3. BM25 (Best Matching 25)
- **Definition**: A probabilistic lexical retrieval algorithm that ranks documents based on the matching query terms appearing in each document.
- **Key Mechanics**:
  1. *Term Frequency (TF)*: How often a query term appears in a document. BM25 applies **saturation**: seeing a word 20 times is not 20 times more important than seeing it once.
  2. *Inverse Document Frequency (IDF)*: Rare terms across the corpus (e.g., "photosynthesis") receive much higher weight than ubiquitous terms (e.g., "system").
  3. *Document Length Normalization*: Penalizes unusually long documents to prevent verbose texts from winning solely because they contain more total words.
- **Why we need it in RAG**: Dense embeddings often struggle with exact keyword matching (acronyms, model codes, serial numbers). BM25 acts as an exact-match safeguard.

---

## 4. Cosine Similarity
- **Definition**: A metric that measures the cosine of the angle between two non-zero vectors in a multi-dimensional space.
- **Formula**:
  $$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Range**: $[-1, 1]$. When vectors are normalized to unit length ($\|\mathbf{u}\|_2 = 1$), cosine similarity simplifies to a simple dot product ($\mathbf{u} \cdot \mathbf{v}$).
- **Interview Takeaway**: Cosine similarity measures *directional alignment* rather than magnitude. A short sentence and a long paragraph expressing the same concept will point in the same direction even if their raw vector lengths differed.

---

## 5. Reciprocal Rank Fusion (RRF)
- **Definition**: An unsupervised, rank-based algorithm that merges the ranked search results from multiple distinct retrieval systems (e.g., BM25 + Dense).
- **Formula**:
  $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $r_m(d)$ is the 1-based rank of document $d$ in system $m$, and $k$ is a constant (typically $k=60$).
- **Why it is superior to Score Blending**:
  BM25 scores and Cosine similarities have incompatible scales and probability distributions. Direct linear combination ($\alpha S_{\text{bm25}} + \beta S_{\text{dense}}$) requires sensitive calibration. RRF uses *only ordinal ranks*, making it completely scale-independent and robust out-of-the-box.

---

## 6. Cross-Encoder
- **Definition**: A transformer architecture that takes both the query and the document concatenated together as a single input sequence: `[CLS] Query [SEP] Document [SEP]`.
- **How it differs from Bi-Encoders**:
  - *Bi-Encoder*: Query and Document are encoded independently into isolated vectors. No cross-attention occurs between query words and document words.
  - *Cross-Encoder*: Self-attention layers compute all-to-all attention between every token of the query and every token of the document simultaneously.
- **Trade-off**: Cross-Encoders are significantly more accurate than Bi-Encoders, but cannot pre-compute embeddings offline. Hence, in HybridRAG, we use Bi-Encoder + BM25 to retrieve the top 10-20 candidates, and the Cross-Encoder only reranks those top candidates.

---

## 7. Grounding
- **Definition**: The technique of restricting an LLM's generation strictly to verified information provided in the prompt context.
- **Why it matters**: Without grounding, large language models hallucinate facts from their pre-training data. Grounding ensures the model says "I don't know based on the provided context" when the answer is not explicitly documented.

---

## 8. Citations & Attribution
- **Definition**: The practice of linking every claim in the generated answer back to the exact source chunk, document name, and page number.
- **Why it matters**: In enterprise and scientific applications, users must be able to click on an answer segment and inspect the underlying raw paragraph to verify factual correctness.

---

## 9. Recall@k
- **Definition**: An information retrieval metric measuring the proportion of relevant documents that appear within the top $k$ positions of the ranked list.
- **Formula**:
  $$\text{Recall@}k = \frac{|\text{Relevant Documents in Top } k|}{|\text{Total Relevant Documents for the Query}|}$$
- **Interpretation**: If a query has 2 relevant chunks in the corpus, and 1 of them appears in your top 5 retrieved results, your $\text{Recall@}5 = \frac{1}{2} = 50\%$.

---

## 10. MRR (Mean Reciprocal Rank)
- **Definition**: An evaluation metric evaluating where the *first* relevant document appears in the ranked results across a set of queries.
- **Formula**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the rank position of the first relevant document for query $i$. If no relevant document is found, the reciprocal rank is 0.
- **Interpretation**:
  - If the first relevant document is at Rank 1, reciprocal rank is $1/1 = 1.0$.
  - If it is at Rank 2, reciprocal rank is $1/2 = 0.5$.
  - If it is at Rank 4, reciprocal rank is $1/4 = 0.25$.
  A high MRR indicates that users almost always find what they are looking for at the very top of the list.
