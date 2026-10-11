"""Multimodal indexing for tables and figures.

Provides ``process_table`` and ``process_figure`` that build text-embeddable
representations per Phase 4:

  Paper: {paper_title} | Type: Table | Section: {section}
  Caption: {caption}
  Context: {' '.join(mentions)}

  {table_markdown}          <- header line (+ separator) repeated for every batch

  Paper: {paper_title} | Type: Figure | Section: {section}
  Caption: {caption}
  Mentions: {' '.join(mentions)}

Pure-text helpers (chunking, overlap, reference filtering) remain in
``src.chunker``. This module handles only visuals.
"""

from __future__ import annotations

import re
from typing import Any

# Markdown table rows with a leading/trailing pipe.
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# Header repeated for every table chunk.
TABLE_BATCH_DATA_ROWS = 15
# Tables with more data rows than this are batched.
TABLE_CHUNK_THRESHOLD = 20


def _parse_markdown_table(table_markdown: str) -> tuple[str, str, list[str]]:
    """Split a Markdown table into (header, separator, data_rows).

    Handles empty / non-tabular input gracefully. If a separator line
    (e.g. ``|---|---|``) cannot be detected the whole input is treated as
    already-formatted markdown with no batched splitting.

    Returns:
        (header, separator, data_rows)
        Any of them may be "" / [] when parsing fails.
    """
    lines = [line.rstrip() for line in (table_markdown or "").strip().splitlines()]
    pipe_lines = [line for line in lines if _PIPE_ROW.match(line)]
    if len(pipe_lines) < 2:
        # No recognisable header+separator; treat whole string as a single block.
        return "", "", []
    header = pipe_lines[0].strip()
    # Separator is normally the second pipe row (|---|---|) but be defensive.
    sep = ""
    for candidate in pipe_lines[1:]:
        if re.match(r"^\s*\|?\s*[-:| ]+\s*\|?\s*$", candidate) or "---" in candidate:
            sep = candidate.strip()
            break
    if not sep:
        sep = pipe_lines[1].strip()
    # Data rows are pipe rows after the separator.
    try:
        sep_index = pipe_lines.index(sep)
    except ValueError:
        sep_index = 1
    data_rows = [line.strip() for line in pipe_lines[sep_index + 1 :]]
    return header, sep, data_rows


def _header_preamble(paper_title: str, section: str, caption: str, mentions: list[str], label: str) -> str:
    """Build the prepended contextual header shared by table/figure chunks."""
    paper_part = f"Paper: {paper_title}" if paper_title else ""
    type_part = f"Type: {label}"
    section_part = f"Section: {section}" if section else ""
    head = " | ".join(part for part in [paper_part, type_part, section_part] if part)
    ctx = " ".join(mentions or [])
    if label == "Table":
        return f"{head}\nCaption: {caption}\nContext: {ctx}" if head else f"Caption: {caption}\nContext: {ctx}"
    return f"{head}\nCaption: {caption}\nMentions: {ctx}" if head else f"Caption: {caption}\nMentions: {ctx}"


def process_table(table_element: dict, paper_title: str) -> list[dict]:
    """Format a schema ``table`` element into embeddable chunks.

    Markdown table handling:
      - ``table_element['content']`` is expected to hold the Markdown table.
      - Header row + separator are repeated at the top of every batch.
      - If the table has ``<= TABLE_CHUNK_THRESHOLD`` data rows it is emitted
        as a single chunk; otherwise data rows are batched in groups of
        ``TABLE_BATCH_DATA_ROWS``.

    Args:
        table_element: one schema.md element with ``type == 'table'``
        paper_title: parent paper's ``title`` for the contextual preamble

    Returns:
        list of chunk dicts (one element when the table is small, multiple
        when batched). Each chunk carries: chunk_id, paper_id, element_id,
        type="table", page, section, title, caption, context, image_path,
        content (embeddable text), raw_text (markdown batch), header.
    """
    paper_id = str(table_element.get("paper_id") or "")
    element_id = str(table_element.get("element_id") or "")
    section = str(table_element.get("section") or "")
    page = table_element.get("page")
    caption = str(table_element.get("caption") or "")
    mentions: list[str] = list(table_element.get("mentions") or [])
    raw_table_data = table_element.get("table_data")
    table_data = raw_table_data if isinstance(raw_table_data, dict) else None
    table_markdown = (table_element.get("content") or "").strip()
    if not table_markdown and table_data:
        headers = [str(value) for value in table_data.get("headers") or []]
        rows = [["" if value is None else str(value) for value in row] for row in table_data.get("rows") or []]
        if headers:
            table_markdown = "| " + " | ".join(headers) + " |\n"
            table_markdown += "| " + " | ".join("---" for _ in headers) + " |\n"
            table_markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows)

    # Fall back: if content is empty (current SPIQA subset), still produce
    # a searchable chunk from the caption/mentions alone; we emit the content
    # as-is so captions without pipe syntax are not dropped.
    has_markdown = bool(table_markdown) and "|" in table_markdown
    preamble = _header_preamble(paper_title, section, caption, mentions, "Table")

    context = " ".join(mentions or [])
    base_fields = {
        "paper_id": paper_id,
        "element_id": element_id,
        "type": "table",
        "page": page,
        "section": section,
        "title": paper_title or "",
        "caption": caption,
        "context": context,
        "image_path": table_element.get("image_path"),
        "source_image": table_element.get("source_image") or "",
        "figure_type": table_element.get("figure_type") or "",
        "mentions": mentions,
        "table_data": table_data,
        "source_url": table_element.get("source_url"),
        "source_location": table_element.get("source_location"),
    }

    if not has_markdown:
        # No tabular content: one chunk whose markdown is the content itself
        # (possibly empty). The preamble still makes the chunk retrievable.
        batch_md = table_markdown
        doc = f"{preamble}\n\n{batch_md}" if batch_md else preamble
        return [{
            **base_fields,
            "chunk_id": f"{paper_id}::{element_id}::t00",
            "chunk_index": 0,
            "n_chunks": 1,
            "header": preamble,
            "raw_text": batch_md,
            "content": doc,
        }]

    header, sep, data_rows = _parse_markdown_table(table_markdown)
    if not data_rows or not header:
        # Un-parseable markdown: keep the original string intact as one chunk.
        doc = f"{preamble}\n\n{table_markdown}"
        return [{
            **base_fields,
            "chunk_id": f"{paper_id}::{element_id}::t00",
            "chunk_index": 0,
            "n_chunks": 1,
            "header": preamble,
            "raw_text": table_markdown,
            "content": doc,
        }]

    batches: list[str] = []
    if len(data_rows) <= TABLE_CHUNK_THRESHOLD:
        batches.append("\n".join([header, sep] + data_rows))
    else:
        for start in range(0, len(data_rows), TABLE_BATCH_DATA_ROWS):
            batch = data_rows[start : start + TABLE_BATCH_DATA_ROWS]
            batches.append("\n".join([header, sep] + batch))

    chunks: list[dict] = []
    for idx, batch_md in enumerate(batches):
        doc = f"{preamble}\n\n{batch_md}"
        chunks.append({
            **base_fields,
            "chunk_id": f"{paper_id}::{element_id}::t{idx:02d}",
            "chunk_index": idx,
            "n_chunks": len(batches),
            "header": preamble,
            "raw_text": batch_md,
            "content": doc,
        })
    return chunks


def process_figure(figure_element: dict, paper_title: str) -> dict | None:
    """Format a schema ``figure`` element into an embeddable chunk.

    Args:
        figure_element: one schema.md element with ``type == 'figure'``
        paper_title: parent paper's ``title`` for the contextual preamble

    Returns:
        chunk dict or ``None`` when the figure carries no caption and no
        mention text.
    """
    paper_id = str(figure_element.get("paper_id") or "")
    element_id = str(figure_element.get("element_id") or "")
    section = str(figure_element.get("section") or "")
    page = figure_element.get("page")
    caption = str(figure_element.get("caption") or "").strip()
    mentions: list[str] = list(figure_element.get("mentions") or [])
    preamble = _header_preamble(paper_title, section, caption, mentions, "Figure")

    if not caption and not mentions:
        return None

    context = " ".join(mentions or [])
    # Spec format: Paper | Type: Figure | Section \nCaption: ...\nMentions: ...
    doc = preamble  # preamble already contains the full spec string for figures

    return {
        "paper_id": paper_id,
        "element_id": element_id,
        "type": "figure",
        "page": page,
        "section": section,
        "title": paper_title or "",
        "caption": caption,
        "context": context,
        "mentions": mentions,
        "image_path": figure_element.get("image_path"),
        "source_image": figure_element.get("source_image") or "",
        "figure_type": figure_element.get("figure_type") or "",
        "source_url": figure_element.get("source_url"),
        "source_location": figure_element.get("source_location"),
        "chunk_id": f"{paper_id}::{element_id}::f00",
        "chunk_index": 0,
        "n_chunks": 1,
        "header": preamble,
        "raw_text": preamble,  # the full figure representation is the preamble itself
        "content": doc,
    }
