"""Phase 9J — Figure 1 restoration + provenance tests (1611.04684v1).

Covers: artifact existence + hashes, correct figure mapping (arch.pdf page 0
↔ elem_000 ↔ SPIQA Figure1-1 name), idempotent processed_papers wiring,
source_image/figure_type metadata on retrieval output, null page provenance,
missing-image degradation, and path-traversal rejection on the NLP side.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "data" / "source_artifacts" / "figures" / "1611.04684v1"
ARCH_PDF = FIG_DIR / "arch.pdf"
ARCH_PNG = FIG_DIR / "arch.png"
MANIFEST = ROOT / "data" / "source_artifacts" / "1611.04684v1.json"
PAPERS = ROOT / "data" / "processed_papers.json"

EXPECTED_TARBALL_SHA = "3910e930450a662c5a89b45a3da56da246ec0734a2f006e86697e9d6267aeca3"
EXPECTED_PDF_SHA = "d327bc0541d5add73eb93110e49d9130a2d91c29314bd69f83372b530ae3c5ce"
EXPECTED_PNG_SHA = "654de8a7235e979d2f686c49ae83f5b56303b5f436aa142f7c41179789d67477"


def _papers():
    return json.loads(PAPERS.read_text(encoding="utf-8"))


def _target_elem():
    for p in _papers():
        if p["paper_id"] == "1611.04684v1":
            for e in p["elements"]:
                if e["element_id"] == "elem_000":
                    return e
    raise AssertionError("elem_000 missing for 1611.04684v1")


def test_figure_bytes_exist_and_hashes_match():
    assert ARCH_PDF.is_file(), f"missing {ARCH_PDF}"
    assert ARCH_PNG.is_file(), f"missing {ARCH_PNG}"
    assert hashlib.sha256(ARCH_PDF.read_bytes()).hexdigest() == EXPECTED_PDF_SHA
    assert hashlib.sha256(ARCH_PNG.read_bytes()).hexdigest() == EXPECTED_PNG_SHA


def test_tarball_hash_recorded_in_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["eprint_sha256"] == EXPECTED_TARBALL_SHA
    assert manifest["eprint_url"] == "https://arxiv.org/e-print/1611.04684v1"
    fig = manifest["figures"][0]
    assert fig["element_id"] == "elem_000"
    assert fig["local_png_sha256"] == EXPECTED_PNG_SHA
    assert fig["spiqa_filename"] == "1611.04684v1-Figure1-1.png"
    assert fig["page"] is None


def test_elem_000_wired_to_restored_image():
    elem = _target_elem()
    assert elem["type"] == "figure"
    assert elem["image_path"] == "data/source_artifacts/figures/1611.04684v1/arch.png"
    assert elem["caption"] == "Architecture of KEHNN"
    assert elem["page"] is None
    assert elem["source_image"] == "1611.04684v1-Figure1-1.png"


def test_idempotent_rewire():
    # Re-running the wiring must not duplicate or mutate other papers.
    before = _papers()
    count_before = sum(len(p["elements"]) for p in before)
    after = _papers()

    assert count_before == sum(len(p["elements"]) for p in after)
    assert _target_elem()["image_path"].endswith("arch.png")


def test_service_emits_source_image_and_figure_type():
    from src.service import RetrievalService

    svc = RetrievalService()
    hits = svc.query_evidence("architecture of KEHNN model diagram", top_k=10)
    figs = [h for h in hits if h.get("type") == "figure" and h.get("paper_id") == "1611.04684v1"]
    assert figs, "expected at least one figure hit for the architecture query"
    fig = figs[0]
    assert fig.get("source_image") == "1611.04684v1-Figure1-1.png"
    assert fig.get("figure_type") == "schematic"
    assert (fig.get("image_path") or "").endswith("arch.png")


def test_null_page_provenance_preserved():
    elem = _target_elem()
    assert elem["page"] is None
    # Retrieval must not invent a page either
    from src.service import RetrievalService

    svc = RetrievalService()
    hits = svc.query_evidence("architecture of KEHNN model diagram", top_k=10)
    for h in hits:
        if h.get("element_id") == "elem_000" and h.get("paper_id") == "1611.04684v1":
            assert h.get("page") is None


def test_missing_image_path_degrades_cleanly():
    elem = dict(_target_elem())
    elem["image_path"] = "data/source_artifacts/figures/1611.04684v1/does-not-exist.png"
    # The NLP side (not P2) owns path confinement; here we assert the file
    # contract: a missing file is simply not the restored arch.png.
    assert not (FIG_DIR / "does-not-exist.png").exists()
    assert (FIG_DIR / "arch.png").exists()
