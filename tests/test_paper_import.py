"""Online paper import tests (P2 venv, mocked HTTP — no network)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_extract_arxiv_id_variants():
    from src.paper_import import extract_arxiv_id

    assert extract_arxiv_id("1611.04684v1") == "1611.04684v1"
    assert extract_arxiv_id("arxiv:1611.04684") == "1611.04684"
    assert extract_arxiv_id("see 1802.07222v1 for details") == "1802.07222v1"
    assert extract_arxiv_id("not-an-id") is None
    assert extract_arxiv_id("") is None


def test_search_parses_atom_feed(monkeypatch: pytest.MonkeyPatch):
    from src import paper_import as pi

    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1611.04684v1</id><title>Knowledge Enhanced Hybrid Neural Network</title>
<summary>We study text matching.</summary><published>2016-11-14T00:00:00Z</published>
<author><name>Yi Tay</name></author>
<link title="pdf" href="https://arxiv.org/pdf/1611.04684v1"/>
</entry></feed>"""

    class FakeResponse:
        def raise_for_status(self): ...

        @property
        def text(self): return feed

    class FakeClient:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return FakeResponse()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "Client", FakeClient)
    hits = pi.search_arxiv("knowledge enhanced", max_results=5)

    assert len(hits) == 1
    assert hits[0].arxiv_id == "1611.04684v1"
    assert hits[0].pdf_url == "https://arxiv.org/pdf/1611.04684v1"
    assert hits[0].source_url == "https://arxiv.org/abs/1611.04684v1"


def test_search_empty_query_rejected():
    from src.paper_import import search_arxiv

    with pytest.raises(ValueError, match="must not be empty"):
        search_arxiv("   ")


def test_download_rejects_disallowed_host():
    from src.paper_import import download_pdf_bytes

    with pytest.raises(ValueError, match="not allowed"):
        download_pdf_bytes("https://evil.example.com/paper.pdf")
    with pytest.raises(ValueError, match="not allowed"):
        download_pdf_bytes("http://arxiv.org/pdf/1611.04684v1")


def test_download_rejects_non_pdf_content(monkeypatch: pytest.MonkeyPatch):
    from src import paper_import as pi

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        content = b"<html>not a pdf</html>"
        def raise_for_status(self): ...

    class FakeClient:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return FakeResponse()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "Client", FakeClient)
    with pytest.raises(ValueError, match="content-type"):
        pi.download_pdf_bytes("https://arxiv.org/pdf/1611.04684v1")


def test_download_rejects_oversize(monkeypatch: pytest.MonkeyPatch):
    from src import paper_import as pi

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = b"%PDF-" + b"0" * (pi.MAX_PDF_BYTES + 1)
        def raise_for_status(self): ...

    class FakeClient:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return FakeResponse()

    import httpx as _httpx

    monkeypatch.setattr(_httpx, "Client", FakeClient)
    with pytest.raises(ValueError, match="exceeds"):
        pi.download_pdf_bytes("https://arxiv.org/pdf/1611.04684v1")


def test_import_rejects_invalid_identifier():
    from src.paper_import import import_arxiv_paper

    with pytest.raises(ValueError, match="not a valid arXiv"):
        import_arxiv_paper("not-an-id", service=None)  # type: ignore[arg-type]


def test_import_duplicate_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    """Same arXiv id twice → paper-scoped replace, no duplicates."""
    from src import paper_import as pi

    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Import Test Paper", fontsize=16)
    import io
    buf = io.BytesIO(); doc.save(buf)
    pdf_bytes = buf.getvalue()

    monkeypatch.setattr(pi, "download_pdf_bytes", lambda url: pdf_bytes)

    from src.service import RetrievalService
    svc = RetrievalService()
    first = pi.import_arxiv_paper("9999.00001v1", svc)
    second = pi.import_arxiv_paper("9999.00001v1", svc)

    assert first["paper_id"] == second["paper_id"]
    assert first["download_bytes"] == len(pdf_bytes)
    hits = svc.query_evidence("Import Test", paper_ids=[first["paper_id"]], top_k=5)
    assert hits and all(h["paper_id"] == first["paper_id"] for h in hits)
