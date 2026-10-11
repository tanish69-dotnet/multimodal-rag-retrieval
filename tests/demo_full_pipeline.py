#!/usr/bin/env python
"""Live End-to-End Demonstration of the Multimodal RAG Retrieval Pipeline (Person 2).

Runs three scenarios without launching a server:
1. INGEST: Ingest a fresh paper from data/processed_papers.json.
2. QA (Person 3 interface): Ask a multimodal question, print top-3 EvidenceItems.
3. COMPARISON (Person 4 interface): Cross-paper comparison query, print structured results.

All outputs strictly conform to schema.md EvidenceItem contract.
"""

import json
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
from src.data_loader import load_local_subset


def tx(v):
    s = "" if v is None else str(v)
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return s.encode("ascii", errors="replace").decode("ascii")


def print_evidence_block(ev, label: str):
    print(f"\n{'='*20} {label} (rank {ev['rank']}) {'='*20}")
    print(f"  paper_id     : {ev['paper_id']}")
    print(f"  element_id   : {ev['element_id']}")
    print(f"  type         : {ev['type']}")
    print(f"  page         : {ev['page']}")
    print(f"  section      : {ev['section']}")
    print(f"  score        : {ev['score']}")
    print(f"  caption      : {tx(ev.get('caption') or '')}")
    print(f"  context      : {tx(ev.get('context') or '')[:120]}")
    snippet = (ev.get("content") or "").replace("\n", " ")
    print(f"  content (snip): {tx(snippet[:160])}")
    print(f"  image_path   : {ev['image_path']}")


def demo_ingest(svc: RetrievalService) -> str:
    """Scenario 1: Ingest one fresh paper."""
    print("\n" + "#" * 78)
    print("# SCENARIO 1: INGESTION (Person 1 -> RetrievalService)")
    print("#" * 78)

    papers, _ = load_local_subset()
    if not papers:
        raise RuntimeError("No papers in processed_papers.json")
    fresh = papers[0]
    pid = fresh["paper_id"]
    print(f"Selected paper: {pid} ({len(fresh['elements'])} elements)")

    n_chunks = svc.ingest_paper(fresh)
    print(f"Ingested {pid} -> {n_chunks} chunks indexed")
    return pid


def demo_qa(svc: RetrievalService, ingested_pid: str):
    """Scenario 2: Person 3 asks a multimodal question."""
    print("\n" + "#" * 78)
    print("# SCENARIO 2: QUESTION ANSWERING (Person 3 <- RetrievalService)")
    print("#" * 78)

    # A realistic multimodal question targeting a table/figure
    question = (
        "What are the BLEU scores reported in the main results table? "
        "Also show the architecture figure if available."
    )
    print(f"\nQUESTION: {question}")

    evidence = svc.query_evidence(question, top_k=5)
    print(f"\nRetrieved {len(evidence)} evidence items (top 5):")

    for i, ev in enumerate(evidence[:3]):
        print_evidence_block(ev, f"EVIDENCE #{i+1}")

    # Verify schema compliance
    required = {"rank", "score", "paper_id", "element_id", "type", "page",
                "section", "content", "caption", "context", "image_path"}
    for ev in evidence:
        missing = required - set(ev.keys())
        assert not missing, f"Missing keys in EvidenceItem: {missing}"
    print("\n>>> Schema.md contract: ALL evidence items VALID ✓")


def demo_comparison(svc: RetrievalService):
    """Scenario 3: Person 4 cross-paper comparison."""
    print("\n" + "#" * 78)
    print("# SCENARIO 3: CROSS-PAPER COMPARISON (Person 4 <- RetrievalService)")
    print("#" * 78)

    # Pick two distinct papers from the loaded corpus
    all_pids = list(svc._ingested_paper_ids)
    if len(all_pids) < 2:
        raise RuntimeError("Need at least 2 ingested papers for comparison")
    paper_ids = all_pids[:2]
    print(f"Comparing papers: {paper_ids}")

    question = "Describe the methodology and model architecture used in each paper."
    print(f"\nQUERY: {question}")

    results = svc.query_comparison(question, paper_ids=paper_ids, k_per_paper=3)

    for pid, hits in results.items():
        print(f"\n{'='*20} PAPER: {pid} ({len(hits)} hits) {'='*20}")
        for ev in hits:
            print_evidence_block(ev, f"rank {ev['rank']}")
        # Verify per-paper schema
        for ev in hits:
            assert ev["paper_id"] == pid, f"Paper ID mismatch in comparison: {ev['paper_id']} != {pid}"
            required = {"rank", "score", "paper_id", "element_id", "type", "page",
                        "section", "content", "caption", "context", "image_path"}
            missing = required - set(ev.keys())
            assert not missing, f"Missing keys in comparison result: {missing}"
    print("\n>>> Schema.md contract: ALL comparison results VALID ✓")


def main() -> int:
    print("=" * 78)
    print("PERSON 2: END-TO-END LIVE DEMONSTRATION")
    print("=" * 78)

    # Spin up the service (loads corpus automatically)
    svc = RetrievalService(auto_load_corpus=True, corpus_limit=15)

    # Scenario 1: Ingestion
    ingested_pid = demo_ingest(svc)

    # Scenario 2: Person 3 QA
    demo_qa(svc, ingested_pid)

    # Scenario 3: Person 4 Comparison
    demo_comparison(svc)

    print("\n" + "=" * 78)
    print("ALL SCENARIOS EXECUTED SUCCESSFULLY")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())