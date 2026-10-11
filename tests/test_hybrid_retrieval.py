"""Phase 5: Hybrid (dense+BM25) + cross-encoder reranking benchmark.

Runs 5 queries mixing semantic and exact keyword/acronym cases. Measures:
- Dense retrieval latency
- BM25 retrieval latency
- RRF fusion latency
- Reranker latency
- Total query latency
Prints top-3 ranked results with rank, element type, cross-encoder score, caption/snippet.
"""

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def tx(v): 
    s="" if v is None else str(v)
    enc=getattr(sys.stdout,'encoding','utf-8') or 'utf-8'
    try: return s.encode(enc,errors='replace').decode(enc,errors='replace')
    except: return s.encode('ascii',errors='replace').decode('ascii')

from src.data_loader import load_local_subset
from src.hybrid_pipeline import HybridPipeline
from src.chunker import chunk_text_element

TOPK_FINAL=5
NUM_Q=5

def synth_text_paper():
    body="Our hybrid model combines BM25 lexical signals with dense embeddings from BGE-small. We report BLEU, ROUGE, and accuracy on standard benchmarks. "*12
    meth="We use RRF to fuse ranks and bge-reranker-base as a cross-encoder to re-rank candidates for better precision. "*10
    return {
        "paper_id":"hybrid_bench_001",
        "title":"Hybrid Lexical+Semantic Retrieval with RRF and Reranking",
        "elements":[
            {"element_id":"h_t0","type":"text","section":"3. Method","page":3,"content":meth},
            {"element_id":"h_t1","type":"text","section":"4. Results","page":4,"content":body},
            {"element_id":"h_tbl0","type":"table","section":"4. Results","page":4,
             "content":"| Model | BLEU | ROUGE-L | Accuracy |\n|---|---|---|---|\n| BM25-only | 24.1 | 47.8 | 68.5 |\n| Dense-only | 27.6 | 51.2 | 74.3 |\n| Hybrid+Rerank | 31.2 | 55.0 | 79.8 |",
             "caption":"Table A: Hybrid achieves best BLEU and accuracy.",
             "mentions":["Table A shows Hybrid+Rerank yields best BLEU and accuracy."]},
            {"element_id":"h_fig0","type":"figure","section":"3.2 Architecture","page":3,
             "caption":"Figure B: Hybrid pipeline overview (BM25 + Dense + RRF + bge-reranker).",
             "mentions":["Figure B illustrates the fusion and reranking stages."]}
        ]
    }

def queries():
    return [
        ("semantic_concept","How does Reciprocal Rank Fusion combine dense and sparse retrieval ranks?"),
        ("exact_model","Which configuration gives the best BLEU and accuracy: BM25-only, Dense-only, or Hybrid+Rerank?"),
        ("acronym_bm25","Find evidence mentioning BLEU ROUGE-L and exact model names in results."),
        ("figure_lookup","Show the hybrid pipeline architecture figure and its caption."),
        ("method_keywords","What cross-encoder reranker is used and what k_rrf constant?"),
    ]

def main():
    print("="*78); print("1) LOAD CORPUS + INDEX (DENSE+BM25)")
    papers,_ = load_local_subset()
    papers.append(synth_text_paper())
    hp = HybridPipeline()
    nd,ns = hp.index_papers(papers, reset=True)
    print(f"indexed: dense_chunks={nd} sparse_chunks={ns}")

    print("\n"+"="*78); print("2) BENCHMARK 5 QUERIES")
    qs = queries()
    for name,q in qs[:NUM_Q]:
        print(f"\n--- [{name}] {q}")
        t0=time.perf_counter()
        res,stats = hp.search(q,k=TOPK_FINAL)
        t_total_real = time.perf_counter()-t0
        print(f"latency: dense={stats['t_dense']:.4f}s sparse={stats['t_sparse']:.4f}s fuse={stats['t_fuse']:.4f}s rerank={stats['t_rerank']:.4f}s | total(tool)={stats['t_total']:.4f}s wall={t_total_real:.4f}s | n_dense={stats['n_dense']} n_sparse={stats['n_sparse']} n_fused={stats['n_fused']}")
        for h in res[:3]:
            snip = (h.get('caption') or h.get('context') or h.get('content') or '').replace('\n',' ')
            print(tx(f"  #{h['rank']} type={h.get('type')} ce={h.get('ce_score')} | {snip[:90]}"))
    print("\n"+"="*78); print("HYBRID PIPELINE OK")
    return 0

if __name__=='__main__':
    raise SystemExit(main())