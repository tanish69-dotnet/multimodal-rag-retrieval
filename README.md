# Multimodal RAG Retrieval Pipeline (Person 2)

> **Academic Research Assistant** — Dense + Sparse Hybrid Retrieval with Cross-Encoder Reranking for Scientific Paper QA

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERSON 2 RETRIEVAL PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
    │  QUERY       │────▶│  QUERY       │────▶│  DENSE           │
    │  ROUTER      │     │  INTENT      │     │  (BGE-small      │
    │  (intent +   │     │  (table/     │     │  en-v1.5)       │
    │  boost)      │     │  figure/     │     └────────┬─────────┘
    └──────────────┘     │  text/       │              │
                         │  general)    │     ┌────────▼─────────┐
                         └──────────────┘     │  SPARSE         │
                                              │  (BM25Okapi)    │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  RRF FUSION      │
                                              │  (k_rrf = 60)    │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  CROSS-ENCODER   │
                                              │  RERANKER        │
                                              │  (bge-reranker-  │
                                              │   base)          │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  CROSS-REF       │
                                              │  EXPANDER        │
                                              │  (Table/Fig      │
                                              │   citations)     │
                                              └────────┬─────────┘
                                                       │
                                    ┌──────────────────┼──────────────────┐
                                    ▼                  ▼                  ▼
                              ┌───────────┐      ┌───────────┐      ┌───────────┐
                              │ PERSON 3  │      │ PERSON 4  │      │  EVIDENCE │
                              │   QA      │      │ COMPARE   │      │  ITEM     │
                              │ /retrieve │      │ /compare  │      │  (schema) │
                              └───────────┘      └───────────┘      └───────────┘
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- Git
- ~4 GB disk space for models & ChromaDB

### 1. Clone & Environment
```bash
git clone <repo-url>
cd multimodal-rag-retrieval

# Activate the existing venv (or create one)
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/macOS

# Verify dependencies
pip install -r requirements.txt
python tests/test_env.py         # Should print "Environment verification PASSED"
```

### 2. Start the FastAPI Server
```bash
# From project root
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
# Server: http://localhost:8000
# Docs:   http://localhost:8000/docs
```

### 3. Run Live Demo (No Server Needed)
```bash
python tests/demo_full_pipeline.py
```

---

## 📚 Teammate Integration Guide

### Person 1 — Paper Ingestion (Parser → Retrieval)

**Option A: Direct Python API (Recommended for batch scripts)**
```python
from src.service import RetrievalService

svc = RetrievalService()  # Loads SPIQA corpus + models

# Person 1's parsed paper (schema.md input contract)
paper = {
    "paper_id": "arxiv_2308.12345",
    "title": "Attention Is All You Need",
    "elements": [
        {
            "element_id": "elem_001",
            "type": "text",
            "section": "3.1 Encoder and Decoder Stacks",
            "page": 3,
            "content": "The encoder is composed of a stack of N = 6 identical layers...",
        },
        {
            "element_id": "elem_002",
            "type": "table",
            "section": "5. Evaluation",
            "page": 7,
            "content": "| Model | BLEU (EN-DE) | BLEU (EN-FR) |\n|---|---|---|\n| Transformer (base) | 27.3 | 38.1 |",
            "caption": "Table 1: Maximum BLEU scores on translation tasks.",
            "mentions": ["As shown in Table 1, our base model beats previous ensemble models."],
        },
        {
            "element_id": "elem_003",
            "type": "figure",
            "section": "3. Model Architecture",
            "page": 3,
            "caption": "Figure 1: The Transformer model architecture.",
            "mentions": ["Figure 1 illustrates the multi-head self-attention mechanisms."],
            "image_path": "data/extracted/arxiv_2308.12345/fig1.png",
        },
    ]
}

chunks = svc.ingest_paper(paper)
print(f"Indexed {chunks} chunks for {paper['paper_id']}")
```

**Option B: REST API (For web/microservice integration)**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "papers": [{
      "paper_id": "arxiv_2308.12345",
      "title": "Attention Is All You Need",
      "elements": [...]
    }]
  }'
```
Response: `{"ingested": 1, "chunks_created": 42, "paper_ids": ["arxiv_2308.12345"]}`

---

### Person 3 — Question Answering (Retrieval → Reasoning)

**Python API**
```python
from src.service import RetrievalService

svc = RetrievalService()  # Reuses already-loaded indices

question = "What are the BLEU scores reported in the main results table?"
evidence = svc.query_evidence(question, top_k=5)

for ev in evidence:
    print(f"[{ev['rank']}] {ev['type'].upper()} p.{ev['page']} | score={ev['score']:.3f}")
    print(f"    {ev['caption']}")
    print(f"    {ev['content'][:120]}...")
    print()
```
**Output** — List of `EvidenceItem` dicts strictly matching `schema.md`:
```json
[
  {
    "rank": 1,
    "score": 0.8328,
    "paper_id": "arxiv_2308.12345",
    "element_id": "elem_002",
    "type": "table",
    "page": 7,
    "section": "5. Evaluation",
    "content": "| Model | BLEU (EN-DE) | BLEU (EN-FR) |\n|---|---|---|\n| Transformer (base) | 27.3 | 38.1 |",
    "caption": "Table 1: Maximum BLEU scores on translation tasks.",
    "context": "As shown in Table 1, our base model beats previous ensemble models.",
    "image_path": null
  }
]
```

**REST API**
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the BLEU scores in the results?", "top_k": 5}'
```

---

### Person 4 — Cross-Paper Comparison Matrix (UI & Analysis)

**Python API**
```python
from src.service import RetrievalService

svc = RetrievalService()
paper_ids = ["arxiv_2308.12345", "arxiv_2203.01234", "arxiv_2106.07890"]

comparison = svc.query_comparison(
    query="Summarize the model architecture and training methodology.",
    paper_ids=paper_ids,
    k_per_paper=3
)

# comparison is a dict: {paper_id: [EvidenceItem, ...]}
for pid, hits in comparison.items():
    print(f"\n=== {pid} ===")
    for h in hits:
        print(f"  [{h['rank']}] {h['type']} | score={h['score']:.3f} | {h['caption']}")
```
**Output** — Per-paper ranked evidence for side-by-side UI tables:
```json
{
  "arxiv_2308.12345": [
    {"rank": 1, "type": "figure", "caption": "Figure 1: Transformer architecture...", ...},
    {"rank": 2, "type": "text", "section": "3.1", "content": "The encoder is composed...", ...}
  ],
  "arxiv_2203.01234": [...]
}
```

**REST API**
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the model architectures across papers.",
    "paper_ids": ["arxiv_2308.12345", "arxiv_2203.01234"],
    "k_per_paper": 3
  }'
```

---

## 📊 Ablation Study & Empirical Results

| Configuration        | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Avg Latency (s) |
|----------------------|----------|----------|-----------|--------|-----------------|
| BM25_Only            | 0.6468   | 0.8587   | 0.9480    | 0.7375 | 0.0013          |
| Dense_Only           | 0.6431   | 0.8810   | 0.9703    | 0.7562 | 0.0451          |
| Hybrid_RRF           | 0.6506   | 0.8885   | 0.9554    | 0.7514 | 0.0472          |
| Dense_Plus_Rerank    | 0.6543   | 0.9071   | 0.9703    | 0.7587 | 1.8543          |
| Hybrid_Plus_Rerank   | 0.6543   | 0.9071   | 0.9554    | 0.7554 | 2.0395          |

*Evaluated on 269 SPIQA test questions (tables/figures). Reranker = `bge-reranker-base`. Candidate pool = top-10.*

**Key Takeaways**
- **BM25** excels at exact keyword matching (BLEU, ROUGE, model names) with negligible latency.
- **Dense** (BGE-small) generalizes better for semantic queries.
- **Hybrid RRF** combines both strengths at near-zero overhead.
- **Cross-Encoder Reranker** boosts Recall@5 by ~2–3 pts but adds ~1.8 s/query (CPU).
- **Full Hybrid + Rerank** achieves best Recall@5 (0.907) for multimodal QA.

---

## 🧪 Testing & Validation

```bash
# Environment sanity
python tests/test_env.py

# Phase 3: Dense retrieval
python tests/test_dense_retrieval.py

# Phase 4: Multimodal (table/figure)
python tests/test_multimodal_retrieval.py

# Phase 5: Hybrid + Rerank
python tests/test_hybrid_retrieval.py

# Phase 6: Ablation benchmark
python src/run_ablation.py

# Phase 7: Research extras (router, cross-ref, per-paper API)
python tests/test_extras.py

# Phase 8: Integration contract checks
python tests/test_integration.py

# Full live demo (all 3 scenarios)
python tests/demo_full_pipeline.py
```

---

## 📁 Project Structure

```
multimodal-rag-retrieval/
├── data/
│   ├── processed_papers.json      # 50 SPIQA papers, 459 elements
│   ├── evaluation_questions.json  # 269 QAs with GT element links
│   └── ablation_results.csv       # Phase 6 results
├── src/
│   ├── service.py                 # RetrievalService (main API)
│   ├── app.py                     # FastAPI server
│   ├── chunker.py                 # Text chunking (400w / 50 overlap)
│   ├── multimodal_processor.py    # Table batching + figure captions
│   ├── dense_retriever.py         # ChromaDB + BGE-small
│   ├── sparse_retriever.py        # BM25Okapi
│   ├── hybrid_pipeline.py         # RRF + CE rerank + router + expander
│   ├── query_router.py            # Intent detection + boosting
│   ├── cross_ref_expander.py      # Table/Figure citation resolution
│   ├── data_loader.py             # SPIQA -> schema.md converter
│   ├── inspect_spiqa.py           # Dataset schema inspection
│   ├── evaluator.py               # Recall@k / MRR metrics
│   └── run_ablation.py            # Ablation runner
├── tests/
│   ├── test_env.py
│   ├── test_dense_retrieval.py
│   ├── test_multimodal_retrieval.py
│   ├── test_hybrid_retrieval.py
│   ├── test_extras.py
│   ├── test_integration.py
│   └── demo_full_pipeline.py
├── schema.md                      # Data contracts (Person 1/3)
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | FastAPI server port |
| `CHROMA_DIR` | `./chroma_db` | Persistent ChromaDB path |
| `HF_TOKEN` | (optional) | HuggingFace token for higher rate limits |

---

## 📄 License

MIT — See `LICENSE` for details.

---

*Built for the Academic Research Assistant project — Person 2 (RAG + Multimodal Retrieval).*