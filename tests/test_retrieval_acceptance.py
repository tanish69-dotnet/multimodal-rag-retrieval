"""Retrieval acceptance: ranking, paper isolation, duplicates, missing visuals.

Uses the in-memory paper path (no model downloads beyond the standard
service fixtures) so the suite stays close to the ingest tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def svc():
    from src.service import RetrievalService

    return RetrievalService(auto_load_corpus=False)


def _paper(paper_id: str, elements: list[dict]) -> dict:
    return {"paper_id": paper_id, "title": f"Title {paper_id}", "elements": elements}


def _text(eid: str, content: str, section: str = "Abstract") -> dict:
    return {
        "element_id": eid, "type": "text", "section": section,
        "page": 1, "content": content,
    }


def _table(eid: str, caption: str, headers: list[str], rows: list[list[str]]) -> dict:
    md = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
    md += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return {
        "element_id": eid, "type": "table", "section": "", "page": 2,
        "content": md, "caption": caption, "mentions": [],
        "table_data": {"headers": headers, "rows": rows},
    }


def _figure(eid: str, caption: str) -> dict:
    return {
        "element_id": eid, "type": "figure", "section": "", "page": 3,
        "content": "", "caption": caption, "mentions": [caption],
    }


def _seed_two_papers(svc):
    svc.ingest_paper(_paper("iso_a", [
        _text("a_t", "KEHNN uses knowledge gates for long-text matching."),
        _table("a_tbl", "Table 1: F1 scores.", ["Model", "F1"], [["KEHNN", "0.81"], ["Base", "0.70"]]),
        _figure("a_fig", "Figure 1: KEHNN architecture."),
    ]))
    svc.ingest_paper(_paper("iso_b", [
        _text("b_t", "A vision transformer segments images with patch tokens."),
        _table("b_tbl", "Table 1: mIoU scores.", ["Model", "mIoU"], [["ViT", "0.55"], ["CNN", "0.50"]]),
        _figure("b_fig", "Figure 1: Patch embedding diagram."),
    ]))


def test_paper_scoping_is_respected(svc):
    _seed_two_papers(svc)

    hits = svc.query_evidence("knowledge gates long-text matching", paper_ids=["iso_a"], top_k=5)

    assert hits
    assert {h["paper_id"] for h in hits} == {"iso_a"}


def test_different_papers_stay_isolated(svc):
    _seed_two_papers(svc)

    per_paper = svc.query_comparison("segmentation scores", paper_ids=["iso_a", "iso_b"], k_per_paper=3)

    assert set(per_paper) == {"iso_a", "iso_b"}
    for pid, hits in per_paper.items():
        assert all(h["paper_id"] == pid for h in hits)


def test_duplicate_ingestion_does_not_grow_index(svc):
    paper = _paper("dup_001", [
        _text("d_t", "Duplicate ingestion must be idempotent."),
        _table("d_tbl", "Table 1.", ["Model", "F1"], [["X", "0.5"]]),
    ])
    svc.ingest_paper(paper)
    before = (svc.pipeline.dense.count(), svc.pipeline.sparse.count)
    svc.ingest_paper(paper)
    after = (svc.pipeline.dense.count(), svc.pipeline.sparse.count)

    assert after[0] <= before[0]
    hits = svc.query_evidence("idempotent duplicate", paper_ids=["dup_001"], top_k=5)
    assert hits and all(h["paper_id"] == "dup_001" for h in hits)


def test_missing_figures_degrade_to_text(svc):
    _seed_two_papers(svc)

    hits = svc.query_evidence("patch embedding diagram architecture", paper_ids=["iso_b"], top_k=6)
    figures = [h for h in hits if h.get("type") == "figure"]

    assert hits
    # Caption-only figures never carry fabricated image bytes.
    for fig in figures:
        assert fig.get("image_b64") in (None, "")


def test_missing_tables_do_not_fabricate_values(svc):
    svc.ingest_paper(_paper("notab_001", [
        _text("n_t", "This paper has prose and no tables about BLEU scores."),
    ]))

    hits = svc.query_evidence("BLEU scores in the results table", paper_ids=["notab_001"], top_k=5)
    tables = [h for h in hits if h.get("type") == "table"]

    assert tables == []
    for hit in hits:
        assert hit.get("table_data") in (None, {})


def test_table_question_retrieves_table_evidence(svc):
    _seed_two_papers(svc)

    hits = svc.query_evidence("Which method has the best F1 score in Table 1?", paper_ids=["iso_a"], top_k=5)

    assert any(h.get("type") == "table" for h in hits)
    tables = [h for h in hits if h.get("type") == "table"]
    assert all(h.get("table_data") for h in tables)
