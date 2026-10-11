"""Section-aware chunking for schema.md elements.

Two roles:
  1. ``chunk_text_element``   -> full-text chunking per spec:
       - target ~350-400 words per chunk with a ~50 word overlap
       - sections matching "References", "Bibliography", or "Acknowledgements"
         (incl. numbered/prefixed variants) are skipped
       - contextual-retrieval header is prepended to every chunk:
             "Paper: {title} | Section: {section}\\n\\n{chunk_text}"
       - element metadata (paper_id, element_id, type="text", page, section)
         is propagated onto every chunk
  2. ``chunk_visual_element`` -> figure/table elements have no long text, so a
       single chunk is emitted from the caption (kept searchable in Phase 3).

``chunk_paper`` applies both to every element of one schema.md paper record and
returns a flat list of chunk dicts ready for embedding/indexing.

Chunk dict fields
-----------------
  chunk_id                : "<paper_id>::<element_id>::cNN"
  paper_id / element_id   : propagated from the element
  type                    : "text" | "figure" | "table"
  page / section          : propagated (page may be null, section may be "")
  title                   : paper title (used in the contextual header)
  header / raw_text       : contextual header and chunk body (without header)
  content                 : FULL embedded document text (header + body)
  caption / source_image / figure_type  : for visual elements
"""

import re
from typing import Any, Optional

#: Target chunk size in words/tokens (spec: ~350-400 words).
DEFAULT_CHUNK_SIZE = 400
#: Words shared between consecutive chunks (spec: ~50).
DEFAULT_OVERLAP = 50

#: Section headings that should never be indexed.
REFERENCE_SECTION_KEYWORDS = {
    "references",
    "reference",
    "bibliography",
    "bibliographies",
    "acknowledgements",
    "acknowledgement",
    "acknowledgments",
    "acknowledgment",
}


def is_reference_section(section: Optional[str]) -> bool:
    """Return True if a section heading is a references/bibliography/ack section.

    Handles plain headings and numbered/prefixed variants by scanning the first
    few word tokens, e.g. "References", "13. References", "& Acknowledgements".
    """
    if not section:
        return False
    tokens = [t for t in re.split(r"[^a-z']+", section.lower()) if t]
    if not tokens:
        return False
    return any(token in REFERENCE_SECTION_KEYWORDS for token in tokens[:3])


def _sliding_word_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word windows (default 400 words / 50 overlap)."""
    if overlap >= chunk_size:
        overlap = max(1, chunk_size - 1)
    step = chunk_size - overlap

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start += step
    return chunks


def _contextual_header(title: str, section: str) -> str:
    """Build the contextual-retrieval header from non-empty fields."""
    parts = [part for part in (f"Paper: {title}" if title else "", f"Section: {section}" if section else "") if part]
    return " | ".join(parts)


def _base_fields(element: dict, title: str) -> dict:
    paper_id = str(element.get("paper_id") or "")
    element_id = str(element.get("element_id") or "")
    fields = {
        "paper_id": paper_id,
        "element_id": element_id,
        "page": element.get("page"),
        "section": element.get("section") or "",
        "title": title,
    }
    for key in ("source_url", "source_location"):
        if element.get(key):
            fields[key] = element[key]
    return fields


def chunk_text_element(
    element: dict,
    title: str = "",
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Chunk one schema text element (section-aware, contextual headers).

    Returns an empty list for reference sections or empty content. Every chunk
    carries the element's ``paper_id``, ``element_id``, ``type="text"``,
    ``page``, and ``section``.
    """
    section = element.get("section") or ""
    if is_reference_section(section):
        return []
    text = (element.get("content") or "").strip()
    if not text:
        return []

    header = _contextual_header(title, section)
    bodies = _sliding_word_chunks(text, chunk_size, overlap)
    base = _base_fields(element, title)

    chunks = []
    for i, body in enumerate(bodies):
        doc_text = f"{header}\n\n{body}" if header else body
        chunks.append({
            **base,
            "chunk_id": f"{base['paper_id']}::{base['element_id']}::c{i:02d}",
            "type": "text",
            "chunk_index": i,
            "n_chunks": len(bodies),
            "header": header,
            "raw_text": body,
            "content": doc_text,   # embedded document text (contextual retrieval)
        })
    return chunks


def chunk_visual_element(element: dict, title: str = "") -> list[dict]:
    """Emit searchable chunk(s) from a figure/table element.

    Delegates to ``src.multimodal_processor`` so tables batch on
    ``content`` (row groups with repeated headers) and figures include
    mentions; captions without mentions are still indexed. Imported lazily
    to avoid a hard import cycle.
    """
    etype = (element.get("type") or "").lower()
    try:
        if etype == "table":
            from src.multimodal_processor import process_table  # noqa: WPS433
            return process_table(element, title)
        from src.multimodal_processor import process_figure  # noqa: WPS433
        chunk = process_figure(element, title)
        return [chunk] if chunk is not None else []
    except ImportError:
        # Standalone fallback when the processor module is absent: index the caption.
        caption = (element.get("caption") or "").strip()
        if not caption:
            return []
        base = _base_fields(element, title)
        orig_type = element.get("type") or "figure"
        header = f"Paper: {title}" if title else ""
        doc_text = f"{header}\n\n{caption}" if header else caption
        return [{
            **base,
            "chunk_id": f"{base['paper_id']}::{base['element_id']}::v00",
            "type": orig_type,
            "chunk_index": 0,
            "n_chunks": 1,
            "header": header,
            "raw_text": caption,
            "content": doc_text,
            "caption": caption,
            "source_image": element.get("source_image") or "",
            "figure_type": element.get("figure_type") or "",
            "mentions": list(element.get("mentions") or []),
            "image_path": element.get("image_path"),
            "context": " ".join(element.get("mentions") or []),
        }]


def chunk_paper(
    paper: dict,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Convert every element of one schema.md paper into chunks.

    Text elements are section-aware chunked; figure/table elements become one
    caption-based chunk each. Elements without any usable text (no content for
    text, no caption for visuals) are skipped.

    Args:
        paper: one schema.md paper record (``paper_id``, ``title``, ``elements``)
        chunk_size: target words per text chunk
        overlap: words shared between consecutive text chunks

    Returns:
        list of chunk dicts (see module docstring).
    """
    paper_id = str(paper.get("paper_id") or "")
    title = paper.get("title") or ""
    chunks: list[dict] = []

    for element in paper.get("elements") or []:
        element = dict(element)
        element.setdefault("paper_id", paper_id)
        etype = (element.get("type") or "").lower()
        if etype == "text":
            chunks.extend(chunk_text_element(element, title, chunk_size=chunk_size, overlap=overlap))
        else:
            chunks.extend(chunk_visual_element(element, title))
    return chunks


def chunk_corpus(
    papers: list[dict],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Chunk a list of schema.md papers into one flat list of chunks."""
    chunks: list[dict] = []
    for paper in papers:
        chunks.extend(chunk_paper(paper, chunk_size=chunk_size, overlap=overlap))
    return chunks
