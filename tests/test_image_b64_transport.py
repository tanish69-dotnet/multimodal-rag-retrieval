"""Phase 9N — trusted-root image_b64 transport tests (P2 venv).

Covers: figure→b64 present with mime, text/table→absent, missing file→None,
traversal outside roots→None, UNC→None, oversize→None, malformed bytes→None,
query_evidence + query_comparison shapes, existing text/table behavior.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_png(path: Path, size: tuple[int, int] = (32, 32)) -> bytes:
    from PIL import Image
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    raw = buf.getvalue()
    path.write_bytes(raw)
    return raw


def test_figure_bytes_returned_with_mime(tmp_path: Path):
    from src.image_transport import image_to_b64, trusted_roots

    target = tmp_path / "fig.png"
    raw = _write_png(target)
    b64, mime = image_to_b64(str(target), roots=[tmp_path])

    assert mime == "image/png"
    assert base64.b64decode(b64) == raw
    assert trusted_roots()  # at least the versioned + temp roots exist as concepts


def test_text_and_table_never_carry_b64():
    # Service wiring only attaches image_b64 for type == "figure" — assert the
    # helper itself is type-agnostic but the service gates on type (covered by
    # test_service_* below emitting no image_b64 for text/table).
    from src.image_transport import image_to_b64

    assert image_to_b64(None) == (None, None)
    assert image_to_b64("") == (None, None)


def test_missing_file_degrades_to_none(tmp_path: Path):
    from src.image_transport import image_to_b64

    assert image_to_b64(str(tmp_path / "does-not-exist.png"), roots=[tmp_path]) == (None, None)


def test_traversal_outside_roots_rejected(tmp_path: Path):
    from src.image_transport import image_to_b64

    outside = tmp_path.parent / "outside.png"
    _write_png(outside)
    # Even though the file exists, it is outside roots=[tmp_path/sub]
    sub = tmp_path / "sub"
    sub.mkdir(exist_ok=True)
    assert image_to_b64(str(outside), roots=[sub]) == (None, None)
    assert image_to_b64("../outside.png", roots=[sub]) == (None, None)


def test_unc_paths_rejected(tmp_path: Path):
    from src.image_transport import image_to_b64

    assert image_to_b64("\\\\server\\share\\x.png", roots=[tmp_path]) == (None, None)
    assert image_to_b64("//server/share/x.png", roots=[tmp_path]) == (None, None)


def test_oversize_image_degrades_to_none(tmp_path: Path):
    from src.image_transport import MAX_IMAGE_BYTES, image_to_b64

    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG\r\n" + b"\x00" * (MAX_IMAGE_BYTES + 1))
    assert image_to_b64(str(big), roots=[tmp_path]) == (None, None)


def test_malformed_bytes_degrade_to_none(tmp_path: Path):
    from src.image_transport import image_to_b64

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png at all, just text")
    assert image_to_b64(str(bad), roots=[tmp_path]) == (None, None)


def test_non_image_suffix_rejected(tmp_path: Path):
    from src.image_transport import image_to_b64

    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert image_to_b64(str(txt), roots=[tmp_path]) == (None, None)


def test_service_query_evidence_carries_b64_only_for_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src import service as svc_module
    from src.service import RetrievalService

    target = tmp_path / "fig.png"
    raw = _write_png(target)

    svc = RetrievalService.__new__(RetrievalService)
    # Bypass the heavy pipeline: stub search to return canned hits
    class FakePipeline:
        def search(self, query, k=5, paper_ids=None, use_router=True, use_expander=True):
            return [
                {
                    "rank": 1,
                    "paper_id": "p1",
                    "element_id": "fig_1",
                    "type": "figure",
                    "page": 1,
                    "section": "",
                    "content": "",
                    "caption": "Figure 1: X",
                    "context": "",
                    "image_path": str(target),
                },
                {
                    "rank": 2,
                    "paper_id": "p1",
                    "element_id": "txt_1",
                    "type": "text",
                    "page": 1,
                    "section": "",
                    "content": "hello",
                    "caption": "",
                    "context": "",
                    "image_path": None,
                },
                {
                    "rank": 3,
                    "paper_id": "p1",
                    "element_id": "tbl_1",
                    "type": "table",
                    "page": 1,
                    "section": "",
                    "content": "| A |",
                    "caption": "Table 1",
                    "context": "",
                    "image_path": None,
                },
            ], {}

    svc.pipeline = FakePipeline()  # type: ignore[attr-defined]
    svc._initialized = True
    svc._ingested_paper_ids = set()
    import src.service as _sm

    monkeypatch.setattr(_sm, "expand_cross_references", lambda *a, **k: ([], {}))

    # Point trusted roots at tmp_path via monkeypatched helper
    import src.image_transport as it

    monkeypatch.setattr(it, "trusted_roots", lambda extra=None: [tmp_path])

    evidence = svc.query_evidence("Figure 1?", top_k=3)
    by_id = {e["element_id"]: e for e in evidence}
    assert by_id["fig_1"]["image_b64"] is not None
    assert by_id["fig_1"]["image_mime"] == "image/png"
    assert base64.b64decode(by_id["fig_1"]["image_b64"]) == raw
    assert "image_b64" not in by_id["txt_1"]
    assert "image_b64" not in by_id["tbl_1"]


def test_service_query_comparison_carries_b64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.service import RetrievalService
    import src.image_transport as it
    import src.service as _sm

    target = tmp_path / "fig.png"
    _write_png(target)

    svc = RetrievalService.__new__(RetrievalService)

    class FakePipeline:
        def retrieve_for_papers(self, query, paper_ids, k_per_paper=3, use_router=True, use_expander=True):
            return {
                "p1": [
                    {
                        "rank": 1,
                        "paper_id": "p1",
                        "element_id": "fig_1",
                        "type": "figure",
                        "page": 1,
                        "section": "",
                        "content": "",
                        "caption": "Figure 1: X",
                        "context": "",
                        "image_path": str(target),
                    }
                ]
            }

    svc.pipeline = FakePipeline()  # type: ignore[attr-defined]
    svc._initialized = True
    monkeypatch.setattr(it, "trusted_roots", lambda extra=None: [tmp_path])
    monkeypatch.setattr(_sm, "expand_cross_references", lambda *a, **k: ([], {}))

    out = svc.query_comparison("Figure 1?", paper_ids=["p1"], k_per_paper=1)
    assert out["p1"][0]["image_b64"] is not None
    assert out["p1"][0]["image_mime"] == "image/png"
