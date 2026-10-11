"""Phase 4 verification: multimodal (text + table + figure) dense retrieval.

1. Index ``data/processed_papers.json`` (tables and figures via
   ``src.multimodal_processor``) plus one synthetic text paper so all three
   schema.md types participate in the same collection.
2. Run three targeted queries:
     Q-text   : methodology / conceptual question  -> expect  text chunks
     Q-table  : "which model achieves best accuracy/BLEU" -> expect table
     Q-figure : "architecture / overview diagram" -> expect figures
3. Print the top-k EvidenceItems showing type, caption, element_id, page,
   section, and score.

Run from the project root:  python tests/test_multimodal_retrieval.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows consoles default to cp1252; keep unicode printable
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def tx(value) -> str:
    text = "" if value is None else str(value)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")

from src.chunker import chunk_text_element  # noqa: E402
from src.data_loader import load_local_subset  # noqa: E402
from src.dense_retriever import DenseRetriever  # noqa: E402

TOP_K = 5
GT_CHECK_TOP_K = 12


def synthetic_text_paper() -> dict:
    """One in-schema paper with real full-text sections (SPIQA has none)."""
    body = (
        "We propose a hybrid retrieval model that combines sparse and dense "
        "representations over scientific papers. The method first indexes every "
        "figure, table, and text paragraph, then fuses lexical and semantic "
        "scores at inference time. Our model achieves a significant accuracy "
        "improvement over the dense-only baseline on the benchmark corpus, "
        "while keeping the index build time near-constant. " * 14
    )
    methodology = (
        "The multimodal retriever embeds table markdown and figure captions "
        "together with surrounding text, then performs k-nearest-neighbour "
        "search against a persistent vector store. We evaluate ranking quality "
        "with recall at ten and mean reciprocal rank, and ablate the effect of "
        "table row batching on long result tables. " * 10
    )
    return {
        "paper_id": "synth_multimodal_001",
        "title": "A Unified Multimodal Retrieval Model for Scientific Papers",
        "elements": [
            {
                "element_id": "elem_text_000",
                "type": "text",
                "section": "3. Method",
                "page": 3,
                "content": methodology,
                "caption": "",
                "mentions": [],
                "image_path": None,
            },
            {
                "element_id": "elem_text_001",
                "type": "text",
                "section": "2. Background and Model",
                "page": 2,
                "content": body,
                "caption": "",
                "mentions": [],
                "image_path": None,
            },
            {
                "element_id": "elem_table_000",
                "type": "table",
                "section": "4. Experiments",
                "page": 6,
                "content": (
                    "| Model | Accuracy | BLEU (EN-DE) | BLEU (EN-FR) |\n"
                    "|---|---|---|---|\n"
                    "| Dense-only baseline | 71.2 | 26.4 | 36.9 |\n"
                    "| Hybrid (sparse + dense) | 78.9 | 29.7 | 40.2 |\n"
                    "| Oracle (multimodal) | 82.4 | 31.1 | 42.8 |\n"
                ),
                "caption": "Table 1: Main results. The hybrid multimodal retriever achieves the best accuracy and BLEU scores.",
                "mentions": [
                    "As shown in Table 1, the hybrid retriever outperforms the dense-only baseline by 7.7 accuracy points.",
                ],
                "image_path": None,
            },
            {
                "element_id": "elem_figure_000",
                "type": "figure",
                "section": "3.1 Architecture",
                "page": 4,
                "content": "",
                "caption": "Figure 2: Overview of the proposed multimodal retrieval architecture.",
                "mentions": [
                    "Figure 2 illustrates the encoder stack, vector store, and fusion layer.",
                ],
                "image_path": "data/extracted/synth_multimodal_001/fig2.png",
            },
        ],
    }


def build_queries() -> list[dict]:
    """Three targeted Phase 4 queries (text / table / figure)."""
    return [
        {
            "name": "TEXT",
            "description": "methodology / conceptual question",
            "query": (
                "What is the fusion strategy used to combine sparse and dense "
                "representation scores in the retrieval model at inference time?"
            ),
            "expect_types": {"text"},
        },
        {
            "name": "TABLE",
            "description": "metrics / results table",
            "query": (
                "Which model achieves the best accuracy and BLEU score in the "
                "results table? Report the hybrid and oracle numbers."
            ),
            "expect_types": {"table"},
        },
        {
            "name": "FIGURE",
            "description": "architecture overview diagram",
            "query": (
                "Find the figure that presents an overview of the multimodal "
                "retrieval architecture with the encoder stack and vector store."
            ),
            "expect_types": {"figure"},
        },
    ]


def fmt_none(value, width):
    text = "n/a" if value in (None, "") else str(value)
    return text[: width - 3] + "..." if len(text) > width else text.ljust(width)


def run_query(retriever, case, debug=True) -> dict:
    hits = retriever.retrieve(case["query"], k=TOP_K)
    seen = {h["type"] for h in hits}
    print(tx(f"query    : {case['query'][:110]}"))
    print(tx(f"expects  : {sorted(case['expect_types'])}  |  retrieved types: {sorted(seen)}"))
    print(f"{'rank':<4} {'score':>6}  {'type':<7} {'page':<5} {'section':<24} {'element_id':<16} caption/snippet")
    for h in hits:
        snippet = (h.get("content") or "").replace("\n", " ")
        cap = h.get("caption") or ""
        snippet = cap or snippet
        print(f"{h['rank']:<4} {fmt_none(h['score'], 6)}  {fmt_none(h['type'], 7)} "
              f"{fmt_none(h['page'], 5)} {fmt_none(h['section'], 24)} {fmt_none(h['element_id'], 16)} {tx(snippet[:52])}")
    return {"hits": hits, "seen": seen}


def main() -> int:
    print("=" * 78)
    print("1) LOAD SUBSET + SYNTHETIC TEXT PAPER")
    print("=" * 78)
    papers, questions = load_local_subset()
    synth = synthetic_text_paper()
    corpus = papers + [synth]

    n_text = sum(1 for p in corpus for e in p["elements"] if (e.get("type") or "") == "text")
    n_table = sum(1 for p in corpus for e in p["elements"] if (e.get("type") or "") == "table")
    n_figure = sum(1 for p in corpus for e in p["elements"] if (e.get("type") or "") == "figure")
    print(f"papers={len(corpus)}  elements: text={n_text} table={n_table} figure={n_figure}")

    print("\n" + "=" * 78)
    print("2) INDEX (text + table + figure) INTO ChromaDB")
    print("=" * 78)
    retriever = DenseRetriever()
    n_chunks = retriever.index_papers(corpus)
    print(f"collection '{retriever.collection_name}' -> {retriever.count()} chunks in {retriever.chroma_dir}")
    assert n_chunks == retriever.count(), "index/collection count mismatch"

    # Per-type metadata check on the stored collection.
    types = {}
    for t in ("text", "table", "figure"):
        r = retriever.collection.get(where={"type": t}, limit=1)
        types[t] = len(r["ids"])
    print(f"stored per type  : {types}")
    assert all(types.values()), "an element type indexed zero chunks"

    # Verify a figure chunk carries image_path + caption metadata end to end.
    fig_doc = retriever.collection.get(where={"element_id": "elem_figure_000"}, limit=1)
    if fig_doc["metadatas"]:
        fm = fig_doc["metadatas"][0]
        print(f"figure metadata   : caption={str(fm.get('caption'))[:40]!r} "
              f"image_path={str(fm.get('image_path'))[:52]!r} context={'<set>' if fm.get('context') else '<empty>'}")

    print("\n" + "=" * 78)
    print(f"3) TARGETED QUERIES (top-{TOP_K})")
    print("=" * 78)
    results = {}
    for case in build_queries():
        print(f"\n--- [{case['name']}] {case['description']} ---")
        results[case["name"]] = run_query(retriever, case)

    # Ground-truth sanity: real SPIQA questions -> exact figure/table elements.
    print("\n" + "=" * 78)
    print(f"4) SPIQA GT IN TOP-{GT_CHECK_TOP_K} (real table/figure questions)")
    print("=" * 78)
    hits_tot = gt_hits = 0
    for qi, q in enumerate(questions[:8]):
        gt = q.get("reference_element_id")
        hits = retriever.retrieve(q["question"], k=GT_CHECK_TOP_K)
        hit_elements = {h["element_id"] for h in hits}
        ok = gt in hit_elements
        hits_tot += len(hits)
        gt_hits += int(ok)
        print(f"  {q['question_id']:<24} GT={gt or '?'} ({q.get('reference_type')}) "
              f"-> {'HIT' if ok else 'miss'}")
    print(f"\n  GT recall@{GT_CHECK_TOP_K} over {len(questions[:8])} questions: {gt_hits}/{len(questions[:8])}")

    for name in ("TEXT", "TABLE", "FIGURE"):
        assert results[name]["seen"], f"{name} query returned no hits"
    print("\nMULTIMODAL RETRIEVAL PIPELINE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())