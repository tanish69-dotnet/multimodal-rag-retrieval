"""Run ablations over SPIQA evaluation questions for Person 2.

Configurations (5):
1. BM25_Only
2. Dense_Only (bge-small-en-v1.5)
3. Hybrid_RRF (Dense+BM25 RRF k_rrf=60, top-15+15 fused, no rerank; evaluate up to 10)
4. Dense_Plus_Rerank (Dense top-10 -> bge-reranker-base)
5. Hybrid_Plus_Rerank (Hybrid RRF top-10 -> bge-reranker-base)

Notes:
- Reranker candidate pool = top-10 to keep CPU evaluation runtime reasonable.
- Relevance = element_id match (and (paper_id,element_id) consistent).
- Corpus = data/processed_papers.json + a small synthetic paper for method keywords if desired? 
  But easier to index the same processed papers + the hybrid_bench synthetic used earlier? 
  To be consistent with prior tests, include the hybrid_bench_001 synthetic paper so method keywords exist.

Outputs: data/ablation_results.csv and Markdown table to stdout.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.data_loader import load_local_subset
from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_pipeline import HybridPipeline, K_RRF, DENSE_TOP, SPARSE_TOP, FUSE_TOP
from src.evaluator import evaluate_questions, save_csv, print_markdown_table
from sentence_transformers import CrossEncoder

QUEST = ROOT / "data" / "evaluation_questions.json"
OUT_CSV = ROOT / "data" / "ablation_results.csv"

# Synthetic to help method-keyword queries
def synth():
    return {
        "paper_id": "hybrid_bench_001",
        "title": "Hybrid Lexical+Semantic Retrieval with RRF and Reranking",
        "elements": [
            {
                "element_id": "h_t0",
                "type": "text",
                "section": "3. Method",
                "page": 3,
                "content": "We use RRF to fuse ranks and bge-reranker-base as a cross-encoder to re-rank candidates for better precision. k_rrf is set to 60 and we rerank the fused top-20.",
            },
            {
                "element_id": "h_t1",
                "type": "text",
                "section": "4. Results",
                "page": 4,
                "content": "Our hybrid model combines BM25 lexical signals with dense embeddings from BGE-small. We report BLEU, ROUGE-L, and accuracy on standard benchmarks.",
            },
            {
                "element_id": "h_tbl0",
                "type": "table",
                "section": "4. Results",
                "page": 4,
                "content": "| Model | BLEU | ROUGE-L | Accuracy |\n|---|---|---|---|\n| BM25-only | 24.1 | 47.8 | 68.5 |\n| Dense-only | 27.6 | 51.2 | 74.3 |\n| Hybrid+Rerank | 31.2 | 55.0 | 79.8 |",
                "caption": "Table A: Hybrid achieves best BLEU and accuracy.",
                "mentions": ["Table A shows Hybrid+Rerank yields best BLEU and accuracy."],
            },
            {
                "element_id": "h_fig0",
                "type": "figure",
                "section": "3.2 Architecture",
                "page": 3,
                "caption": "Figure B: Hybrid pipeline overview (BM25 + Dense + RRF + bge-reranker).",
                "mentions": ["Figure B illustrates the fusion and reranking stages."],
            },
        ],
    }


def load_questions(limit=None):
    with open(QUEST, encoding="utf-8") as f:
        qs = json.load(f)
    return qs[:limit] if limit else qs


def main():
    print("=" * 78)
    print("ABLATION: load corpus + build indices")
    print("=" * 78)
    papers, _ = load_local_subset()
    papers.append(synth())
    # build shared
    dense = DenseRetriever()
    sparse = SparseRetriever()
    nd = dense.index_papers(papers, reset=True)
    ns = sparse.index_papers(papers, reset=True)
    hp = HybridPipeline(dense=dense, sparse=sparse, ce_model="BAAI/bge-reranker-base")
    ce = CrossEncoder("BAAI/bge-reranker-base")
    print(f"done: dense={nd} sparse={ns}")

    qs = load_questions()
    results = {}

    # 1. BM25_Only
    print("\n[1/5] BM25_Only...")
    def bm25_only(q, pidlist=None):
        t0 = time.perf_counter()
        h = sparse.retrieve(q, pidlist, k=10)
        lat = time.perf_counter() - t0
        # normalize to EvidenceItem-like shape (dense has rank; bm25 gives rank_bm25)
        out = []
        for x in h:
            out.append(
                {
                    "rank": x.get("rank_bm25") or len(out) + 1,
                    "score": x.get("bm25_score"),
                    "paper_id": x.get("paper_id"),
                    "element_id": x.get("element_id"),
                    "type": x.get("type"),
                    "page": x.get("page"),
                    "section": x.get("section"),
                    "content": x.get("content"),
                    "caption": x.get("caption"),
                    "context": x.get("context"),
                }
            )
        return out, lat

    results["BM25_Only"] = evaluate_questions(qs, bm25_only)

    # 2. Dense_Only
    print("[2/5] Dense_Only...")
    def dense_only(q, pidlist=None):
        t0 = time.perf_counter()
        h = dense.retrieve(q, pidlist, k=10)
        lat = time.perf_counter() - t0
        return h, lat

    results["Dense_Only"] = evaluate_questions(qs, dense_only)

    # 3. Hybrid_RRF (no rerank, evaluate top-10)
    print("[3/5] Hybrid_RRF...")
    def hybrid_rrf(q, pidlist=None):
        t0 = time.perf_counter()
        dens = dense.retrieve(q, pidlist, k=15)
        spar = sparse.retrieve(q, pidlist, k=15)
        fused = hp._rrf_fuse(dens, spar)[:10]
        lat = time.perf_counter() - t0
        # assign rank
        for i, x in enumerate(fused, 1):
            x["rank"] = i
        return fused, lat

    results["Hybrid_RRF"] = evaluate_questions(qs, hybrid_rrf)

    # 4. Dense_Plus_Rerank (top-10)
    print("[4/5] Dense_Plus_Rerank...")
    def dense_rerank(q, pidlist=None):
        t0 = time.perf_counter()
        cands = dense.retrieve(q, pidlist, k=10)
        if not cands:
            return [], time.perf_counter() - t0
        pairs = [(q, c["content"] or "") for c in cands]
        scores = ce.predict(pairs)
        items = [dict(c) for c in cands]
        for it, s in zip(items, scores):
            it["ce_score"] = float(s)
        items.sort(key=lambda x: x["ce_score"], reverse=True)
        for i, it in enumerate(items[:10], 1):
            it["rank"] = i
        lat = time.perf_counter() - t0
        return items[:10], lat

    results["Dense_Plus_Rerank"] = evaluate_questions(qs, dense_rerank)

    # 5. Hybrid_Plus_Rerank (top-10 fused)
    print("[5/5] Hybrid_Plus_Rerank...")
    def hybrid_rerank(q, pidlist=None):
        t0 = time.perf_counter()
        dens = dense.retrieve(q, pidlist, k=15)
        spar = sparse.retrieve(q, pidlist, k=15)
        fused = hp._rrf_fuse(dens, spar)[:10]
        if not fused:
            return [], time.perf_counter() - t0
        pairs = [(q, c["content"] or "") for c in fused]
        scores = ce.predict(pairs)
        items = [dict(c) for c in fused]
        for it, s in zip(items, scores):
            it["ce_score"] = float(s)
        items.sort(key=lambda x: x["ce_score"], reverse=True)
        for i, it in enumerate(items[:10], 1):
            it["rank"] = i
        lat = time.perf_counter() - t0
        return items[:10], lat

    results["Hybrid_Plus_Rerank"] = evaluate_questions(qs, hybrid_rerank)

    save_csv(results, OUT_CSV)
    print("\n" + "=" * 78)
    print("ABLATION RESULTS (Markdown)")
    print_markdown_table(results)
    print(f"\nSaved: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())