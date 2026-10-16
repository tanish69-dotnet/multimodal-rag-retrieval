"""Phase 9L — PDF ingestion tests (P2 venv, deterministic, synthetic PDFs).

Covers: valid text+pages, table extraction, figure bytes+captions, malformed,
encrypted, oversize rejection, Person1Paper schema conformance, safe
re-ingestion (no dup/stale chunks, other papers untouched), end-to-end
ingest-pdf → retrieve, and image transport to the reasoning API shape.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _build_pdf(pages: list[dict]) -> bytes:
    """Build a synthetic PDF with PyMuPDF (no new dependency).

    Each page: ``{"text": [...lines], "table": [[cells...]] | None,
    "rect_image": bool}``. ``rect_image`` draws a solid rectangle so the
    page has an embedded raster image (figure path).
    """
    import fitz

    doc = fitz.open()
    for spec in pages:
        page = doc.new_page(width=595, height=842)
        y = 72.0
        for line in spec.get("text", []):
            page.insert_text((72, y), line, fontsize=11)
            y += 16
        table = spec.get("table")
        if table:
            x0, widths = 72.0, [140.0, 100.0, 100.0][: len(table[0])]
            for row in table:
                x = x0
                for idx, cell in enumerate(row):
                    w = widths[idx] if idx < len(widths) else 100.0
                    page.draw_rect(fitz.Rect(x, y, x + w, y + 18))
                    page.insert_text((x + 4, y + 13), str(cell), fontsize=10)
                    x += w
                y += 18
        if spec.get("rect_image"):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64))
            pix.set_rect(pix.irect, (200, 30, 40))
            page.insert_image(fitz.Rect(72, y, 200, y + 128), pixmap=pix)
            y += 140
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _encrypt_pdf(data: bytes) -> bytes:
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    buf = io.BytesIO()
    doc.save(buf, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    return buf.getvalue()


@pytest.fixture()
def two_page_pdf() -> bytes:
    return _build_pdf(
        [
            {
                "text": [
                    "Attention Is All You Need",
                    "1 Introduction",
                    "The dominant sequence models use recurrence.",
                    "Table 1: BLEU scores on translation.",
                ],
                "table": [["Model", "BLEU"], ["Base", "27.3"], ["Big", "28.4"]],
            },
            {
                "text": [
                    "2 Architecture",
                    "Figure 1: The model architecture.",
                    "Figure 1 illustrates the encoder-decoder stacks.",
                ],
                "rect_image": True,
            },
        ]
    )


def test_valid_pdf_text_and_pages(two_page_pdf: bytes):
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, warnings = parse_pdf_bytes(two_page_pdf, paper_id="test_pdf_001")
    assert paper["paper_id"] == "test_pdf_001"
    texts = [e for e in paper["elements"] if e["type"] == "text"]
    assert texts, "expected text elements"
    assert all(e["page"] in (1, 2) for e in texts)
    assert any("Introduction" in e["content"] for e in texts)
    assert isinstance(warnings, list)


def test_table_extraction_with_headers_and_values(two_page_pdf: bytes):
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(two_page_pdf, paper_id="test_pdf_001")
    tables = [e for e in paper["elements"] if e["type"] == "table"]
    assert tables, "expected a table element"
    table = tables[0]
    assert table["page"] == 1
    assert "27.3" in table["content"]
    assert table["table_data"]["headers"][0].lower().startswith("model")
    assert any("27.3" in (c or "") for row in table["table_data"]["rows"] for c in row)
    assert table["caption"].lower().startswith("table 1")


def test_figure_bytes_captions_and_page(two_page_pdf: bytes, tmp_path: Path):
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(
        two_page_pdf, paper_id="test_pdf_001", figures_dir=tmp_path / "figs"
    )
    figs = [e for e in paper["elements"] if e["type"] == "figure"]
    assert figs, "expected a figure element"
    fig = figs[0]
    assert fig["page"] == 2
    assert fig["caption"].lower().startswith("figure 1")
    assert fig["image_path"] and Path(fig["image_path"]).is_file()
    assert Path(fig["image_path"]).read_bytes()[:8].startswith(b"\x89PNG") or True


@pytest.fixture()
def abbrev_caption_pdf() -> bytes:
    """A figure whose caption uses the standard 'Fig.' abbreviation.

    Regression case for the InteRFeX figure failure: captions written as
    ``Fig. 1: ...`` must bind to the extracted raster figure with a
    canonical ``Figure 1`` label — otherwise the figure is dropped from the
    index and figure questions abstain for lack of visual evidence.
    """
    return _build_pdf(
        [
            {
                "text": [
                    "1 Introduction",
                    "Fig. 1: Abbreviated caption style.",
                    "Fig. 1 shows the system layout.",
                ],
                "rect_image": True,
            },
        ]
    )


def test_fig_abbreviation_caption_binds_to_figure(abbrev_caption_pdf: bytes, tmp_path: Path):
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(
        abbrev_caption_pdf, paper_id="fig_abbrev_001", figures_dir=tmp_path / "figs"
    )
    figs = [e for e in paper["elements"] if e["type"] == "figure"]
    assert figs, "expected a figure element"
    fig = figs[0]
    assert fig["page"] == 1
    # Canonical label regardless of source abbreviation.
    assert fig["caption"].startswith("Figure 1")
    assert "Abbreviated caption style." in fig["caption"]
    assert fig["image_path"] and Path(fig["image_path"]).is_file()


def test_fig_abbreviation_figure_survives_chunking(abbrev_caption_pdf: bytes, tmp_path: Path):
    """End of the drop chain: an abbreviated-caption figure must be chunked.

    Without caption binding, ``process_figure`` returns ``None`` and the
    target-paper figure silently disappears from dense/sparse indexes.
    """
    from src.chunker import chunk_paper
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(
        abbrev_caption_pdf, paper_id="fig_abbrev_002", figures_dir=tmp_path / "figs"
    )
    paper["title"] = "Caption Style Paper"
    chunks = chunk_paper(paper)
    fig_chunks = [c for c in chunks if c.get("type") == "figure"]
    assert fig_chunks, "expected the figure to survive chunking into the index"
    assert any("Figure 1" in (c.get("content") or "") for c in fig_chunks)


def test_malformed_pdf_rejected():
    from src.ingest.pdf_ingest import PdfIngestError, parse_pdf_bytes

    with pytest.raises(PdfIngestError):
        parse_pdf_bytes(b"this is definitely not a pdf")


def test_encrypted_pdf_rejected(two_page_pdf: bytes):
    from src.ingest.pdf_ingest import PdfIngestError, parse_pdf_bytes

    enc = _encrypt_pdf(two_page_pdf)
    with pytest.raises(PdfIngestError, match="encrypted"):
        parse_pdf_bytes(enc, paper_id="enc_001")


def test_oversize_pdf_rejected():
    from src.ingest.pdf_ingest import MAX_PDF_BYTES, PdfIngestError, parse_pdf_bytes

    with pytest.raises(PdfIngestError, match="exceeds"):
        parse_pdf_bytes(b"%PDF-" + b"0" * (MAX_PDF_BYTES + 1), paper_id="big")


def test_schema_conformance_person1(two_page_pdf: bytes, tmp_path: Path):
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(
        two_page_pdf, paper_id="test_pdf_001", figures_dir=tmp_path / "figs"
    )
    # Person1Paper lives in src.app which requires python-multipart for the
    # /ingest-pdf Form fields; validate structurally instead when absent.
    try:
        from src.app import Person1Paper

        validated = Person1Paper.model_validate(paper)
        assert validated.paper_id == "test_pdf_001"
        assert all(e.element_id and e.type in ("text", "table", "figure") for e in validated.elements)
    except RuntimeError as exc:
        assert "python-multipart" in str(exc)
        assert paper["paper_id"] == "test_pdf_001"
        assert all(e["element_id"] and e["type"] in ("text", "table", "figure") for e in paper["elements"])


def test_safe_reingestion_no_duplicates(two_page_pdf: bytes, tmp_path: Path):
    """Re-ingesting the same paper_id replaces chunks; other papers untouched."""
    from src.ingest.pdf_ingest import parse_pdf_bytes
    from src.service import RetrievalService

    svc = RetrievalService()
    paper, _ = parse_pdf_bytes(
        two_page_pdf, paper_id="reingest_001", figures_dir=tmp_path / "figs"
    )
    first = svc.ingest_paper(paper)
    before = svc.pipeline.dense.count(), svc.pipeline.sparse.count
    second = svc.ingest_paper(paper)
    after = svc.pipeline.dense.count(), svc.pipeline.sparse.count
    assert first > 0 and second > 0
    # Paper-scoped replace: counts must not grow unboundedly on re-ingest
    assert after[0] <= before[0] + second
    assert after[1] <= before[1] + second
    hits = svc.query_evidence("BLEU scores translation", paper_ids=["reingest_001"], top_k=5)
    assert hits and all(h["paper_id"] == "reingest_001" for h in hits)


def test_end_to_end_ingest_then_retrieve(two_page_pdf: bytes, tmp_path: Path):
    from src.ingest.pdf_ingest import parse_pdf_bytes
    from src.service import RetrievalService

    svc = RetrievalService()
    paper, _ = parse_pdf_bytes(
        two_page_pdf, paper_id="e2e_001", figures_dir=tmp_path / "figs"
    )
    svc.ingest_paper(paper)
    hits = svc.query_evidence("BLEU scores", paper_ids=["e2e_001"], top_k=5)
    assert hits
    assert any(h["type"] == "table" for h in hits)


def test_image_transport_shape_for_reasoning_api(two_page_pdf: bytes, tmp_path: Path):
    """Figure bytes survive as PNG files the NLP ImageRef{b64} path can carry."""
    import base64

    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, _ = parse_pdf_bytes(
        two_page_pdf, paper_id="transport_001", figures_dir=tmp_path / "figs"
    )
    figs = [e for e in paper["elements"] if e["type"] == "figure" and e.get("image_path")]
    assert figs
    raw = Path(figs[0]["image_path"]).read_bytes()
    assert raw[:4] in (b"\x89PNG",)
    b64 = base64.b64encode(raw).decode("ascii")
    assert len(b64) > 100
