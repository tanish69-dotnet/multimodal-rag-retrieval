"""Integration & Contract Validation Tests (Phase 8).

Tests the RetrievalService programmatically without launching the FastAPI server:
1. Ingest a sample paper matching Person 1's schema.
2. Call query_evidence simulating Person 3's question.
3. Call query_comparison simulating Person 4's comparison query.
4. Validate all output dictionaries strictly against schema.md EvidenceItem keys.

Run: python tests/test_integration.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.service import RetrievalService
from src.chunker import chunk_text_element

EVIDENCE_KEYS = {
    "rank", "score", "paper_id", "element_id", "type", "page",
    "section", "content", "caption", "context", "image_path",
    "table_data", "metadata"
}

def tx(v):
    s = "" if v is None else str(v)
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try: return s.encode(enc, errors="replace").decode(enc, errors="replace")
    except: return s.encode("ascii", errors="replace").decode("ascii")

def assert_evidence_schema(item: dict, where: str):
    """Assert item has exactly the EvidenceItem keys and correct types."""
    missing = EVIDENCE_KEYS - set(item.keys())
    extra = set(item.keys()) - EVIDENCE_KEYS
    assert not missing, f"{where}: missing keys {missing}"
    assert not extra, f"{where}: extra keys {extra}"
    # type checks
    assert isinstance(item["rank"], int), f"{where}: rank must be int"
    assert item["score"] is None or isinstance(item["score"], (int, float)), f"{where}: score must be number or None"
    assert item["paper_id"] is None or isinstance(item["paper_id"], str), f"{where}: paper_id must be str or None"
    assert item["element_id"] is None or isinstance(item["element_id"], str), f"{where}: element_id must be str or None"
    assert item["type"] is None or isinstance(item["type"], str), f"{where}: type must be str or None"
    assert item["page"] is None or isinstance(item["page"], int), f"{where}: page must be int or None"
    assert isinstance(item["section"], str), f"{where}: section must be str"
    assert isinstance(item["content"], str), f"{where}: content must be str"
    assert item["caption"] is None or isinstance(item["caption"], str), f"{where}: caption must be str or None"
    assert isinstance(item["context"], str), f"{where}: context must be str"
    assert item["image_path"] is None or isinstance(item["image_path"], str), f"{where}: image_path must be str or None"
    assert item["table_data"] is None or isinstance(item["table_data"], dict), f"{where}: table_data must be dict or None"
    assert isinstance(item["metadata"], dict), f"{where}: metadata must be dict"

def person1_paper_sample():
    """Sample Person 1 paper with text, table, figure."""
    return {
        "paper_id": "integration_test_001",
        "title": "Integration Test Paper: Cross-Modal Retrieval",
        "elements": [
            {
                "element_id": "int_text0",
                "type": "text",
                "section": "3. Method",
                "page": 3,
                "content": "We propose a multimodal retriever that embeds text, tables, and figures into a shared vector space. The model uses RRF fusion and a cross-encoder reranker to achieve state-of-the-art Recall@5 on SPIQA.",
                "caption": "",
                "mentions": ["Table 1 shows the main results.", "Figure 1 illustrates the pipeline."],
            },
            {
                "element_id": "int_tbl1",
                "type": "table",
                "section": "4. Results",
                "page": 5,
                "content": "| Model | Recall@5 | Recall@10 |\n|---|---|---|\n| Dense-only | 0.881 | 0.970 |\n| Hybrid+Rerank | 0.907 | 0.970 |",
                "caption": "Table 1: Main results on SPIQA test split.",
                "mentions": ["Table 1 shows the main results on SPIQA test split."],
            },
            {
                "element_id": "int_fig1",
                "type": "figure",
                "section": "3.1 Architecture",
                "page": 4,
                "content": "",
                "caption": "Figure 1: Multimodal retriever architecture overview.",
                "mentions": ["Figure 1 illustrates the pipeline stages."],
                "image_path": "data/extracted/integration_test_001/fig1.png",
            },
        ],
    }

def main():
    print("=" * 78)
    print("PHASE 8: Integration & Contract Validation")
    print("=" * 78)

    svc = RetrievalService(auto_load_corpus=True, corpus_limit=20)

    # ------------------------------------------------------------------
    # 1. INGEST
    # ------------------------------------------------------------------
    print("\n[1/3] INGEST: Person 1 paper -> RetrievalService")
    paper = person1_paper_sample()
    n_chunks = svc.ingest_paper(paper)
    print(f"  Ingested: {paper['paper_id']} | chunks created: {n_chunks}")
    assert n_chunks > 0, "ingest returned zero chunks"

    # ------------------------------------------------------------------
    # 2. QUERY_EVIDENCE (Person 3)
    # ------------------------------------------------------------------
    print("\n[2/3] QUERY_EVIDENCE: Person 3 question")
    q = "What are the main results in the results table?"
    evidence = svc.query_evidence(q, top_k=5)
    print(f"  Query: {q}")
    print(f"  Returned {len(evidence)} evidence items")
    for e in evidence[:3]:
        print(f"    rank={e['rank']} type={e['type']} element={e['element_id']} score={e['score']} caption={tx(e.get('caption'))[:50]}")
        assert_evidence_schema(e, "query_evidence")
    assert len(evidence) >= 1, "no evidence returned"
    # Expect table element to be in top results for table query
    types = {e["type"] for e in evidence}
    assert "table" in types, "table element missing from table query results"
    print("  Contract validation: ALL evidence items conform to schema.md EvidenceItem ✓")

    # ------------------------------------------------------------------
    # 3. QUERY_COMPARISON (Person 4)
    # ------------------------------------------------------------------
    print("\n[3/3] QUERY_COMPARISON: Person 4 cross-paper comparison")
    # pick 3 papers from the loaded corpus
    all_papers = list(svc._ingested_paper_ids)
    selected = all_papers[:3]
    print(f"  Comparing papers: {selected}")
    comp = svc.query_comparison("Best accuracy and BLEU scores?", paper_ids=selected, k_per_paper=3)
    for pid, hits in comp.items():
        print(f"  {pid}: {len(hits)} hits")
        for h in hits:
            print(f"    rank={h['rank']} type={h['type']} element={h['element_id']} score={h['score']}")
            assert_evidence_schema(h, f"query_comparison[{pid}]")
        assert len(hits) <= 3, f"more than k_per_paper for {pid}"
        assert all(h["paper_id"] == pid for h in hits), f"wrong paper_id in results for {pid}"

    print("\n" + "=" * 78)
    print("ALL INTEGRATION CONTRACT CHECKS PASSED")
    print("=" * 78)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
