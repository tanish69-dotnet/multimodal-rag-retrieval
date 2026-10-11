"""Phase 7 validation: Query Router, Cross-Ref Expander, and Per-Paper API."""

import sys
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
from src.hybrid_pipeline import HybridPipeline
from src.query_router import classify_intent, boost_matching_type
from src.cross_ref_expander import expand_cross_references, ExpanderConfig
from src.chunker import chunk_text_element


def tx(v):
    s = "" if v is None else str(v)
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return s.encode("ascii", errors="replace").decode("ascii")


def synth_paper():
    """Paper with a text chunk citing Table 1 and Figure 2."""
    return {
        "paper_id": "expander_test_001",
        "title": "Cross-Reference Expansion Test Paper",
        "elements": [
            {
                "element_id": "t_chunk0",
                "type": "text",
                "section": "3. Experiments",
                "page": 3,
                "content": "As shown in Table 1, our model outperforms baselines. Figure 2 illustrates the architecture.",
            },
            {
                "element_id": "tbl1",
                "type": "table",
                "section": "3. Experiments",
                "page": 3,
                "content": "| Model | Acc |\n|---|---|\n| Ours | 92.3 |",
                "caption": "Table 1: Main results comparison.",
                "mentions": ["Table 1 shows our model achieves the highest accuracy."],
            },
            {
                "element_id": "fig2",
                "type": "figure",
                "section": "3. Experiments",
                "page": 3,
                "caption": "Figure 2: Architecture overview.",
                "mentions": ["Figure 2 illustrates the overall model architecture."],
            },
        ],
    }


def load_corpus():
    papers, _ = load_local_subset()
    papers.append(synth_paper())
    return papers


def test_query_router():
    print("\n" + "=" * 78)
    print("TEST 1: Query Router Intent Classification & Boosting")
    print("=" * 78)

    queries = [
        ("table", "Which model achieves the best BLEU score in the results table?"),
        ("figure", "Show me the architecture diagram of the model."),
        ("text", "Explain the methodology and training loss function."),
        ("general", "What is the capital of France?"),
    ]

    for expected, q in queries:
        intent = classify_intent(q)
        print(f"  '{q[:50]}...' -> intent={intent} {'✓' if intent == expected else '✗'}")
        assert intent == expected, f"expected {expected}, got {intent} for '{q}'"

    # Test boosting changes scores and ordering
    hits = [
        {"element_id": "e1", "type": "table", "score": 0.7, "ce_score": 0.6},
        {"element_id": "e2", "type": "figure", "score": 0.65, "ce_score": 0.7},
        {"element_id": "e3", "type": "text", "score": 0.6, "ce_score": 0.5},
    ]
    boosted = boost_matching_type(hits.copy(), "table")
    # table element should have increased score
    assert boosted[0]["score"] > 0.7, f"table score not boosted: {boosted[0]['score']}"
    assert boosted[0]["element_id"] == "e1", "table not ranked first after boost"
    print("  Boosting: table element score increased and ranked first ✓")

    print("QUERY ROUTER: PASS")


def test_cross_ref_expander():
    print("\n" + "=" * 78)
    print("TEST 2: Cross-Reference Expansion (Table 1 / Figure 2 resolution)")
    print("=" * 78)

    paper = synth_paper()
    pid = paper["paper_id"]
    elements = paper["elements"]

    # Simulate a hit list with a text chunk citing Table 1 and Figure 2
    text_hit = {
        "paper_id": pid,
        "element_id": "t_chunk0",
        "type": "text",
        "content": "As shown in Table 1, our model outperforms baselines. Figure 2 illustrates the architecture.",
        "section": "3. Experiments",
        "page": 3,
    }

    expanded = expand_cross_references(
        [text_hit],
        get_paper_elements=lambda p: elements if p == pid else [],
        config=ExpanderConfig(max_expansions=4, append=True),
    )

    types = [h.get("type") for h in expanded]
    eids = [h.get("element_id") for h in expanded]
    print(f"  Original types: {types}")
    print(f"  Element IDs   : {eids}")

    # Should have original text + table + figure
    assert "text" in types, "original text chunk missing"
    assert "table" in types, "table element not expanded"
    assert "figure" in types, "figure element not expanded"

    # Check flag
    cross_ref_items = [h for h in expanded if h.get("expanded_via_cross_ref")]
    assert len(cross_ref_items) == 2, f"expected 2 cross-ref expansions, got {len(cross_ref_items)}"
    for cr in cross_ref_items:
        print(f"  Expanded: {cr['element_id']} ({cr['type']}) via trigger '{cr['cross_ref_trigger']}' ✓")

    print("CROSS-REF EXPANDER: PASS")


def test_retrieve_for_papers():
    print("\n" + "=" * 78)
    print("TEST 3: Per-Paper API (retrieve_for_papers)")
    print("=" * 78)

    papers = load_corpus()
    dense = DenseRetriever()
    sparse = SparseRetriever()
    dense.index_papers(papers, reset=True)
    sparse.index_papers(papers, reset=True)

    hp = HybridPipeline(dense=dense, sparse=sparse, ce_model="BAAI/bge-reranker-base")

    # pick 3 distinct papers
    selected = [p["paper_id"] for p in papers[:3]]
    print(f"  Querying papers: {selected}")

    per_paper = hp.retrieve_for_papers(
        "What is the main result or accuracy?",
        paper_ids=selected,
        k_per_paper=3,
        use_router=True,
        use_expander=False,
    )

    for pid in selected:
        hits = per_paper.get(pid, [])
        print(f"  {pid}: {len(hits)} hits")
        for h in hits:
            print(f"    rank={h['rank']} type={h['type']} element={h['element_id']} score={h.get('score') or h.get('ce_score')}")
        assert len(hits) <= 3, f"more than k_per_paper results for {pid}"
        assert all(h.get("paper_id") == pid for h in hits), f"wrong paper_id in results for {pid}"

    print("PER-PAPER API: PASS")


def main():
    test_query_router()
    test_cross_ref_expander()
    test_retrieve_for_papers()
    print("\n" + "=" * 78)
    print("ALL EXTRAS TESTS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())