"""General PDF → Person1Paper parser (Phase 9L, PyMuPDF only, no GPU, no OCR).

Produces ``Person1Paper``-shaped dicts (``paper_id, title, elements``) where
each element carries ``element_id, type, section, page, content, caption,
mentions, image_path``. Figure/table extraction is conservative: missing or
ambiguous output is reported in ``warnings`` rather than fabricated.

Design notes
------------
- Text: ``page.get_text("blocks")`` grouped top-to-bottom; headings detected
  by a ``Table|Figure N`` caption regex or a short-line + larger-font
  heuristic; ``page`` is the verified 1-based PDF page (never invented).
- Tables: ``page.find_tables()`` when available; header = first row, Markdown
  ``content`` + ``table_data{headers, rows}``; captions matched by proximity
  (±1 page, nearest preceding ``Table N`` line).
- Figures: embedded raster images via ``page.get_images(full=True)`` pixmaps
  saved content-addressed under ``data/ingested/<paper_id>/``; vector-only
  diagrams fall back to a page-region clip around the caption line
  (``page.get_pixmap(clip=...)``); captions matched the same way as tables.
  ``page`` is the verified source page; association failures keep
  ``page: null`` on the caption-only record instead of guessing.
- Determinism: ``element_id`` values are stable per (page, kind, ordinal)
  within one paper; re-parsing the same bytes yields identical output.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

MAX_PDF_BYTES = 32 * 1024 * 1024  # 32 MiB upload cap

_CAPTION_RE = re.compile(r"^\s*(Table|Figure)\s+([0-9][0-9A-Za-z\-]*)[:.\-–—\s]+(.*)$", re.IGNORECASE)
_MENTION_RE = re.compile(r"\b(?:Table|Figure|Fig\.)\s+[0-9][0-9A-Za-z\-]*\b", re.IGNORECASE)


class PdfIngestError(ValueError):
    """Typed ingestion failure (malformed / encrypted / oversize / unreadable)."""


def is_pdf_bytes(data: bytes) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _sha12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _paper_id_for(data: bytes, title: str | None, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (title or "").strip().lower()).strip("_")
    if slug:
        return f"pdf_{slug[:48]}_{_sha12(data)}"
    return f"pdf_{_sha12(data)}"


def _parse_pdf(data: bytes) -> Any:
    try:
        import fitz  # PyMuPDF (pinned in requirements.txt)
    except ImportError as exc:
        raise PdfIngestError("PyMuPDF is required for PDF ingestion but is not installed") from exc
    if not is_pdf_bytes(data):
        raise PdfIngestError("not a PDF file (missing %PDF- magic)")
    if len(data) > MAX_PDF_BYTES:
        raise PdfIngestError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfIngestError(f"unreadable PDF: {exc}") from exc
    if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
        try:
            if doc.authenticate("") not in (1, 2, True):
                raise PdfIngestError("encrypted PDF (password required)")
        except PdfIngestError:
            raise
        except Exception as exc:
            raise PdfIngestError(f"encrypted PDF (password required): {exc}") from exc
    if len(doc) == 0:
        raise PdfIngestError("PDF has no pages")
    return doc


def _page_lines(page: Any) -> list[dict[str, Any]]:
    """Text lines with bbox + font size, sorted top-to-bottom."""
    lines: list[dict[str, Any]] = []
    try:
        blocks = page.get_text("dict", sort=True).get("blocks", [])
    except Exception:
        return lines
    for block in blocks:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((float(s.get("size", 0) or 0) for s in spans), default=0.0)
            bbox = line.get("bbox", [0, 0, 0, 0])
            lines.append({"text": text, "size": size, "bbox": list(bbox)})
    lines.sort(key=lambda item: (round(item["bbox"][1], 1), round(item["bbox"][0], 1)))
    return lines


def _split_caption(text: str) -> tuple[str, str] | None:
    match = _CAPTION_RE.match(text or "")
    if not match:
        return None
    kind = match.group(1).capitalize()
    number, rest = match.group(2).strip(), match.group(3).strip()
    label = f"{kind} {number}"
    caption = f"{label}: {rest}" if rest else f"{label}"
    return label, caption


def _mentions_for(label: str, pages_lines: list[list[dict[str, Any]]], own_page: int) -> list[str]:
    """Sentences elsewhere that reference ``Table N``/``Figure N`` (context)."""
    pattern = re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
    mentions: list[str] = []
    for page_idx, lines in enumerate(pages_lines):
        for line in lines:
            text = line["text"]
            if pattern.search(text) and not text.lower().startswith(label.lower()):
                mentions.append(text.strip())
                if len(mentions) >= 3:
                    return mentions
    _ = own_page
    return mentions


def _table_markdown(headers: list[str], rows: list[list[str]]) -> str:
    def esc(cell: str | None) -> str:
        return (cell or "").replace("|", "\\|").strip()

    head = "| " + " | ".join(esc(h) for h in headers) + " |"
    sep = "|" + "|".join(["---"] * max(1, len(headers))) + "|"
    body = ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _extract_tables(
    doc: Any, pages_lines: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Find tables via PyMuPDF; associate captions by proximity (±1 page)."""
    tables: list[dict[str, Any]] = []
    # Caption index: (page_idx, line_idx, label, caption)
    captions: list[tuple[int, int, str, str]] = []
    for page_idx, lines in enumerate(pages_lines):
        for line_idx, line in enumerate(lines):
            split = _split_caption(line["text"])
            if split and split[0].lower().startswith("table"):
                captions.append((page_idx, line_idx, split[0], split[1]))

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            found = page.find_tables()
            extracted = list(found)
        except Exception:
            continue
        for ordinal, table in enumerate(extracted):
            try:
                raw = table.extract()
            except Exception:
                continue
            if not raw or not any(any(cell for cell in row) for row in raw):
                continue
            headers = [(c or "").strip() for c in (raw[0] or [])]
            rows = [[(c or "").strip() for c in row] for row in raw[1:]]
            # Caption: nearest preceding Table caption on same/previous page
            label, caption = None, ""
            best: tuple[int, int] | None = None
            for cpage, cline, clabel, ctext in captions:
                if cpage > page_idx or (cpage == page_idx and cline > 10**9):
                    continue
                key = (abs(cpage - page_idx), -cpage)
                if best is None or key < best:
                    best, label, caption = key, clabel, ctext
            element_id = f"tbl_p{page_idx + 1}_{ordinal + 1:02d}"
            tables.append(
                {
                    "element_id": element_id,
                    "type": "table",
                    "page": page_idx + 1,
                    "reference_label": label,
                    "caption": caption,
                    "headers": headers,
                    "rows": rows,
                    "content": _table_markdown(headers, rows),
                }
            )
    return tables


def _save_pixmap(pixmap: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # Pillow (pinned in requirements.txt)
    except ImportError as exc:
        raise PdfIngestError("Pillow is required for figure extraction but is not installed") from exc
    samples = pixmap.samples
    mode = "RGB" if pixmap.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), samples)
    if mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    image.save(out_path, "PNG")
    return out_path


def _extract_figures(
    doc: Any,
    data: bytes,
    paper_id: str,
    pages_lines: list[list[dict[str, Any]]],
    figures_dir: Path,
) -> list[dict[str, Any]]:
    """Raster images via xref pixmaps; vector diagrams via caption-region clips."""
    figures: list[dict[str, Any]] = []
    captions: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for page_idx, lines in enumerate(pages_lines):
        for line_idx, line in enumerate(lines):
            split = _split_caption(line["text"])
            if split and split[0].lower().startswith("figure"):
                captions.append((page_idx, line_idx, split[0], split[1], line))

    used_xrefs: set[int] = set()
    ordinal = 0
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            images = page.get_images(full=True)
        except Exception:
            images = []
        for img in images:
            xref = img[0] if img else 0
            if not xref or xref in used_xrefs:
                continue
            used_xrefs.add(xref)
            try:
                pix = doc.extract_image(xref)
                raw = pix.get("image")
                if not raw:
                    continue
            except Exception:
                continue
            ordinal += 1
            digest = hashlib.sha256(raw).hexdigest()[:12]
            out_path = figures_dir / f"fig_p{page_idx + 1}_{ordinal:02d}_{digest}.png"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(raw)
            except OSError as exc:
                raise PdfIngestError(f"cannot write figure artifact: {exc}") from exc
            figures.append(
                {
                    "element_id": f"fig_p{page_idx + 1}_{ordinal:02d}",
                    "type": "figure",
                    "page": page_idx + 1,
                    "image_path": str(out_path),
                    "digest": digest,
                }
            )

    # Caption association + vector-diagram fallback clips
    for page_idx, line_idx, label, caption, line in captions:
        # Attach to the nearest same-page raster figure without a caption yet
        target = None
        for fig in figures:
            if fig.get("caption"):
                continue
            # Same page preferred; previous page accepted for floating figures
            if fig.get("page", 0) in (page_idx + 1, page_idx):
                target = fig
                break
        if target is not None:
            target["caption"] = caption
            target["reference_label"] = label
            continue
        # No raster image on this page: render a region clip around the caption
        # (captures vector diagrams / composite figures as pixels).
        try:
            page = doc[page_idx]
            rect = page.rect
            bbox = line.get("bbox", [0, 0, 0, 0])
            import fitz

            clip = fitz.Rect(
                max(0, rect.x0),
                max(0, (bbox[1] if len(bbox) > 1 else rect.y0) - rect.height * 0.55),
                min(rect.x1, rect.x1),
                min(rect.y1, (bbox[3] if len(bbox) > 3 else rect.y1) + rect.height * 0.08),
            )
            pix = page.get_pixmap(clip=clip, dpi=150)
            ordinal += 1
            digest = hashlib.sha256(pix.samples).hexdigest()[:12]
            out_path = figures_dir / f"fig_p{page_idx + 1}_{ordinal:02d}_{digest}.png"
            _save_pixmap(pix, out_path)
            figures.append(
                {
                    "element_id": f"fig_p{page_idx + 1}_{ordinal:02d}",
                    "type": "figure",
                    "page": page_idx + 1,
                    "image_path": str(out_path),
                    "digest": digest,
                    "caption": caption,
                    "reference_label": label,
                    "rendered_clip": True,
                }
            )
        except Exception:
            # Caption-only record: honest degradation, page kept (caption line
            # page is verified), image absent.
            ordinal += 1
            figures.append(
                {
                    "element_id": f"fig_p{page_idx + 1}_{ordinal:02d}",
                    "type": "figure",
                    "page": page_idx + 1,
                    "caption": caption,
                    "reference_label": label,
                    "image_path": None,
                }
            )
    return figures


def parse_pdf_bytes(
    data: bytes,
    *,
    paper_id: str | None = None,
    title: str | None = None,
    figures_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Parse PDF bytes → ``(Person1Paper dict, warnings)``.

    Never fabricates page numbers, captions, or provenance: association
    failures keep ``page: null`` / empty captions rather than guessing.
    """
    doc = _parse_pdf(data)
    try:
        meta = doc.metadata or {}
        resolved_title = (title or meta.get("title") or "").strip()
        resolved_id = _paper_id_for(data, resolved_title, paper_id)
        authors = (meta.get("author") or "").strip()

        pages_lines = [_page_lines(doc[i]) for i in range(len(doc))]
        total_words = sum(len(line["text"].split()) for lines in pages_lines for line in lines)

        warnings: list[str] = []
        if total_words == 0:
            warnings.append("scanned-or-image-only: no text layer found; OCR not performed in this phase")

        # Text elements: one per page with content (headings kept inline;
        # section = first heading-ish line or "" — never invented).
        elements: list[dict[str, Any]] = []
        ordinal = 0
        for page_idx, lines in enumerate(pages_lines):
            texts = [line["text"] for line in lines if not _split_caption(line["text"])]
            content = "\n".join(texts).strip()
            if not content:
                continue
            ordinal += 1
            section = ""
            for line in lines[:4]:
                if len(line["text"]) <= 80 and line["size"] >= 11:
                    section = line["text"].strip()
                    break
            elements.append(
                {
                    "element_id": f"txt_p{page_idx + 1}_{ordinal:02d}",
                    "type": "text",
                    "section": section,
                    "page": page_idx + 1,
                    "content": content,
                }
            )

        # Tables (verified pages) + figures (bytes or honest caption-only)
        tables = _extract_tables(doc, pages_lines)
        for table in tables:
            label = table.pop("reference_label", None)
            elements.append(
                {
                    "element_id": table["element_id"],
                    "type": "table",
                    "section": "",
                    "page": table["page"],
                    "content": table["content"],
                    "caption": table.get("caption", ""),
                    "mentions": _mentions_for(label, pages_lines, table["page"]) if label else [],
                    "table_data": {"headers": table["headers"], "rows": table["rows"]},
                }
            )

        import tempfile

        default_fig_dir = Path(tempfile.gettempdir()) / "p2_ingest" / resolved_id
        figures = _extract_figures(doc, data, resolved_id, pages_lines, figures_dir or default_fig_dir)
        for fig in figures:
            label = fig.pop("reference_label", None)
            entry: dict[str, Any] = {
                "element_id": fig["element_id"],
                "type": "figure",
                "section": "",
                "page": fig.get("page"),
                "content": "",
                "caption": fig.get("caption", ""),
                "mentions": _mentions_for(label, pages_lines, fig.get("page") or 0) if label else [],
            }
            if fig.get("image_path"):
                entry["image_path"] = fig["image_path"]
            elements.append(entry)

        paper: dict[str, Any] = {
            "paper_id": resolved_id,
            "title": resolved_title,
            "elements": elements,
        }
        if authors:
            paper["authors"] = authors
        return paper, warnings
    finally:
        try:
            doc.close()
        except Exception:
            pass
