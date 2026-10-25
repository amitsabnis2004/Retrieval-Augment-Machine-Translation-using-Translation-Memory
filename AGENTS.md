# AGENTS.md — Retrieval-Augmented Low-Resource Machine Translation (RAMT)

> **Project Reference & Context File**  
> This file acts as the persistent single source of truth for AI agents and human collaborators working on the Capstone research project.

---

## 1. Project Overview & Institutional Metadata

* **Project Title:** Retrieval-Augmented Low-Resource Machine Translation System Using Translation Memory
* **Institution:** GITAM School of CSE, GITAM (Deemed to be University), Bengaluru, Karnataka, India
* **Department:** Department of Computer Science and Systems Engineering (CSSE)
* **Batch:** CSSE-08
* **Project Guide:** Mr. Gunti Spandan (Assistant Professor)
* **Primary Language Pair:** **English → Kannada (`en → kn`)**
  * *Secondary / Extensibility Target Languages:* Assamese, Bodo, Manipuri, Mizo, Kokborok, Santali (Indic regional languages).

---

## 2. Research Problem & Motivation

### The Problem
* **Data Scarcity:** Many Indian regional languages lack large-scale, high-quality parallel bilingual datasets.
* **Parametric Bottleneck:** Traditional Neural Machine Translation (NMT) models encode all translation knowledge solely within static network weights.
  * They struggle with rare words, regional idioms, and domain-specific terminology (e.g., agriculture, legal, government, medicine).
  * Updating model knowledge requires expensive retraining or fine-tuning, which introduces the risk of catastrophic forgetting.
* **Limitations of Alternative Paradigms:**
  * **Token-Level $k\text{NN-MT}$:** Queries an external datastore at every single decoding token step, causing prohibitive inference latency (often 100x slower) and immense RAM overhead.
  * **Pure Large Language Model (LLM) Prompting:** Prone to hallucinations, lacks token-level grammatical accuracy in low-resource Indic scripts, and requires massive compute/API costs.

### The Proposed Solution
Develop a **Sentence-Level Retrieval-Augmented Machine Translation (RAMT)** system utilizing **Translation Memory (TM)**. The system dynamically retrieves verified bilingual sentence pairs and provides them as contextual grounding to a compact, specialized Indic translation model at inference time.

---

## 3. Team Roster & Module Ownership

| Member | Registration No. | Role | Core Modules & Deliverables |
| :--- | :--- | :--- | :--- |
| **Amit A Sabnis** | 2023002731 | **Team Lead & System Architect** | Overall RAMT architecture, baseline NMT configuration, LoRA/PEFT adaptation, end-to-end integration pipeline, technical coordination |
| **Gautam Kumar** | 2023002846 | **Translation Memory & Retrieval Lead** | TM database schema & storage, sentence embedding pipeline (LaBSE), FAISS vector index (quantized HNSW), top-$k$ retrieval & latency optimization |
| **Niranjan Sudhakar** | 202300**** | **Data Pipeline & Baseline MT Lead** | English → Kannada corpus collection, normalization, tokenization, train/val/test splits, TM datastore source data curation, baseline NMT benchmarking |
| **Sujit Lakshman B** | 2023001514 | **Evaluation, Testing & Interface Lead** | Evaluation pipeline (`chrF++`, `COMET-22`, `BLEU`), targeted test suites (rare words, long sentences, domain terms), demo API & interactive web interface |

---

## 4. System Architecture & Technical Specifications

> Detailed visual flow diagrams for both sub-systems are available in [architecture_diagrams.md](file:///d:/University/Capstone/docs/architecture_diagrams.md).

```
                       +-----------------------------------+
                       | User Input (English Source Text)  |
                       +-----------------+-----------------+
                                         |
                                         v
                         +---------------+---------------+
                         |   Text Preprocessing/Cleaning |
                         +---------------+---------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
   +-----------------------+                           +-----------------------+
   | BM25 Lexical Search   |                           | LaBSE Semantic Embed  |
   | (Exact / Term Match)  |                           | (Universal Vector)    |
   +-----------+-----------+                           +-----------+-----------+
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | FAISS Quantized HNSW Vector Database |
                      | (Translation Memory Bilingual Pairs) |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      |   Candidate TM Sentences Retrieval   |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | Cross-Encoder Reranker & Filtering   |
                      | Threshold-Based Negative Rejection   |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | Source-Side Context Concatenation    |
                      | <src> TM_en <tgt> TM_kn <sep> Query  |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | Generative NMT Backbone              |
                      | IndicTrans2 Distilled 200M (LoRA)    |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | Final Kannada Translation Output     |
                      +------------------+-------------------+
                                         |
                                         v
                      +------------------+-------------------+
                      | Web UI / REST API / Evaluation Suite |
                      +--------------------------------------+
```

### Core Technical Components:
1. **Translation Memory (TM) Datastore:**
   * Aligned bilingual pairs: `(ID, source_text_en, target_text_kn, domain, embedding_vector)`.
   * Sources: AI4Bharat Samanantar, Bharat Parallel Corpus Collection (BPCC), FLORES-200, OPUS, domain corpora (agriculture, legal, governance).
2. **Hybrid Retrieval Strategy:**
   * **Sparse:** BM25 / fuzzy matching for entity and terminology preservation.
   * **Dense:** Multilingual dual-encoder ([LaBSE](https://arxiv.org/abs/2007.01852)) mapping source English and target Kannada into a shared universal semantic space.
3. **Quantized HNSW Index (FAISS):**
   * Hierarchical Navigable Small World (HNSW) graph for high recall and sub-millisecond lookups.
   * Memory optimization via 8-bit signed integer byte quantization (reducing RAM footprint by 75%) with SIMD distance calculations.
4. **Noise Rejection & Cross-Encoder Reranking:**
   * **Context-Awareness Rate (CAR) Mitigation:** Low-resource models over-rely on external context and degrade when given irrelevant retrieved segments.
   * Cross-encoder assigns similarity confidence; segments below confidence threshold $\tau$ are dropped (**Negative Rejection**), falling back gracefully to pure parametric NMT.
5. **Generative Backbone & In-Context Integration:**
   * **Base Model:** `ai4bharat/indictrans2-en-indic-dist-200m` (18-layer encoder, 18-layer decoder, ~211M parameters).
   * **Prompt Formatting:** Source-side concatenation using delimiter tokens:
     ```text
     Context: [SRC] <TM_English> [TGT] <TM_Kannada> [INPUT] <Input_English>
     ```
   * **Parameter-Efficient Fine-Tuning (PEFT):** Low-Rank Adaptation (LoRA) with rank $r=16$, $\alpha=32$ applied to attention projection layers (~0.4% trainable weights), avoiding catastrophic forgetting.
6. **Evaluation Metrics:**
   * **`chrF++`:** Primary metric for agglutinative Indian morphology (character n-grams correlate best with human judgments).
   * **`COMET-22`:** Cross-lingual neural metric for semantic fidelity.
   * **`BLEU` (SacreBLEU):** Standard lexical n-gram benchmark metric.
   * **Targeted Challenge Sets:** Accuracy on rare words, agricultural/technical terms, and complex long sentences.

---

## 5. Key Benchmark Datasets & Literature References

### Primary Datasets:
* **Samanantar (IIT Madras / AI4Bharat):** Largest public parallel corpus for Indian languages.
* **BPCC (Bharat Parallel Corpus Collection):** Preprocessing and benchmark dataset from IndicTrans2.
* **IN22 Suite:** IN22-Gen (1,024 sentences) and IN22-Conv (1,503 multi-turn dialogues) evaluation benchmarks.
* **FLORES-200:** Standard global evaluation benchmark for low-resource translation.

### Foundation Literature (Stored in `docs/papers/`):
1. **IndicTrans2:** Gala et al. (TMLR 2023) — *Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages*.
2. **ANVITA:** J et al. (WMT 2025) — *A Multi-pronged Approach for Enhancing Machine Translation of Extremely Low-Resource Indian Languages*.
3. **Transfer Learning for Indic MT:** Wei et al. (WMT 2024) — *Machine Translation Advancements of Low-Resource Indian Languages by Transfer Learning*.
4. **Levenshtein Transformers RAMT:** Bouthors et al. (EMNLP 2023) — *Towards Example-Based NMT with Multi-Levenshtein Transformers*.
5. **RAGtrans Benchmark:** Wang & Meng et al. (EMNLP 2025) — *Retrieval-Augmented Machine Translation with Unstructured Knowledge*.
6. **LaBSE:** Feng et al. (ACL 2022) — *Language-agnostic BERT Sentence Embedding*.

---

## 6. Proposed Repository & Codebase Directory Layout

```text
Capstone/
├── AGENTS.md                                # This reference context file
├── docs/                                    # Project documentation, reviews, presentations, papers
│   ├── papers/                              # Foundational research papers (PDFs)
│   └── review1/                             # Review 1 presentations, reports, and Gantt charts
├── data/
│   ├── raw/                                 # Raw downloaded parallel corpora (Samanantar, FLORES-200)
│   ├── processed/                           # Tokenized, cleaned, and split datasets (train/val/test)
│   └── tm_store/                            # Translation Memory database and FAISS index files
├── src/
│   ├── data_pipeline/                       # Ingestion, cleaning, Unicode normalization, tokenization
│   ├── embeddings/                          # LaBSE / Sentence-Transformer embedding generation
│   ├── tm_indexer/                          # FAISS HNSW quantization, indexing, and SQLite metadata
│   ├── retriever/                           # Hybrid search (BM25 + Dense) and Cross-Encoder reranker
│   ├── models/                              # IndicTrans2 loader, LoRA adapters, inference pipeline
│   ├── evaluation/                          # SacreBLEU, chrF++, COMET-22, and challenge set evaluation
│   └── app/                                 # Demo API (FastAPI) and Interactive UI (Streamlit/Web)
├── notebooks/                               # Prototyping, data exploration, and visualization notebooks
├── requirements.txt                         # Python dependencies
└── README.md                                # Project summary and instructions
```

---

## 7. Phased Implementation Roadmap (Gantt Milestones)

* **Phase 1 (Aug – Oct):**
  * Problem Identification & Literature Survey *(Completed — Review 1)*
  * Architecture Design & Tool Selection *(Completed — Review 1)*
  * Dataset Collection, Cleaning, and Baseline NMT Setup *(Current Phase)*
* **Phase 2 (Nov – Dec):**
  * Translation Memory Database Creation & FAISS Quantized HNSW Indexing
  * Hybrid Retrieval Engine & Threshold Rejection Implementation
  * Source-Side Concatenation & LoRA Fine-Tuning on IndicTrans2
* **Phase 3 (Jan – Feb):**
  * Automated Benchmarking (`chrF++`, `COMET-22`, `BLEU`) & Error Analysis
  * Robustness Testing (Noise sensitivity, rare word accuracy)
  * Web Demo / API Application Layer Build
* **Phase 4 (Mar):**
  * End-to-End System Integration & Latency Optimization
  * Final Project Report, Thesis Documentation & Final Viva Presentation

---

## 8. Agent Instructions & Coding Guidelines

* **Scripting Environment:** Python 3.10+ (Windows PowerShell environment).
* **Dependencies:** `torch`, `transformers`, `sentence-transformers`, `faiss-cpu` / `faiss-gpu`, `sacrebleu`, `evaluate`, `rank_bm25`, `fastapi`, `uvicorn`, `streamlit`.
* **Reproducibility:** Fix random seeds across PyTorch and NumPy (`seed=42`).
* **Path Conventions:** Use forward slashes or `os.path.join` / `pathlib.Path` to maintain cross-platform compatibility.
* **Documentation:** Preserve all comments, math formulations, and docstrings in code modules. Keep benchmark results logged in structured JSON/CSV format.
