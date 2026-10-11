"""Restore paper text and structured Table 4 from the versioned arXiv HTML.

The SPIQA test split contains captions and QA annotations, but not article text
or table cells.  This module fetches the authoritative versioned HTML for
``1611.04684v1`` and merges only the missing material into the local target
paper record.  The existing SPIQA element IDs and source image names remain
unchanged.

Run from the Person 2 project root::

    python -m src.restore_target_paper

The source HTML, hash manifest, and restored corpus snapshots are preserved
under ``data/`` by the separate repository's explicit ignore exceptions.
ChromaDB and other generated indexes remain ignored. The URL, paper identity,
and merge rules are kept in this module and documented in
``PHASE9H_SOURCE.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

PAPER_ID = "1611.04684v1"
SOURCE_URL = "https://arxiv.org/html/1611.04684v1"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_PATH = DATA_DIR / "processed_papers.json"
SOURCE_DIR = DATA_DIR / "source_artifacts"
SOURCE_HTML_PATH = SOURCE_DIR / f"{PAPER_ID}.html"
SOURCE_META_PATH = SOURCE_DIR / f"{PAPER_ID}.json"


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    parent: "_Node | None" = None
    children: list["_Node | str"] = field(default_factory=list)


class _TreeParser(HTMLParser):
    """Small dependency-free HTML tree sufficient for arXiv's stable classes."""

    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Node(tag.lower(), {key: value or "" for key, value in attrs}, self._stack[-1])
        self._stack[-1].children.append(node)
        if tag.lower() not in self._VOID:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self._VOID and len(self._stack) > 1:
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        wanted = tag.lower()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == wanted:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def _walk(node: _Node) -> Iterable[_Node]:
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _walk(child)


def _classes(node: _Node) -> set[str]:
    return set(node.attrs.get("class", "").split())


def _has_class(node: _Node, name: str) -> bool:
    return name in _classes(node)


def _descendants(node: _Node, *, tag: str | None = None, class_name: str | None = None) -> list[_Node]:
    return [
        child
        for child in _walk(node)
        if (tag is None or child.tag == tag)
        and (class_name is None or _has_class(child, class_name))
    ]


def _clean_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _clean_table_cell(value: str) -> str:
    """Keep source values while removing HTML-renderer math spacing."""
    value = _clean_text(value)
    return re.sub(r"\bR\s*(\d+)\s*@\s*(\d+)\b", r"R\1@\2", value)


def _text(node: _Node) -> str:
    """Extract readable text, using MathML alt text for formulas."""
    if node.tag in {"script", "style", "noscript"}:
        return ""
    if node.tag == "math" and node.attrs.get("alttext"):
        return _clean_text(node.attrs["alttext"])
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, _Node):
            value = _text(child)
        else:
            value = child
        if value:
            parts.append(value)
    return _clean_text(" ".join(parts))


def _first_text(node: _Node, tags: set[str]) -> str:
    for candidate in _walk(node):
        if candidate.tag in tags:
            value = _text(candidate)
            if value:
                return value
    return ""


def _inside(node: _Node, predicate) -> bool:
    current: _Node | None = node
    while current is not None:
        if predicate(current):
            return True
        current = current.parent
    return False


def _nearest_section(node: _Node) -> _Node | None:
    current = node.parent
    while current is not None:
        if current.tag == "section" and any(_has_class(current, name) for name in ("ltx_section", "ltx_subsection", "ltx_subsubsection")):
            return current
        current = current.parent
    return None


def _section_title(section: _Node | None) -> str:
    if section is None:
        return ""
    for candidate in _walk(section):
        if candidate.tag in {"h2", "h3", "h4", "h5"}:
            return _text(candidate)
    return ""


def _is_references(title: str) -> bool:
    return bool(re.search(r"\b(?:references?|bibliograph(?:y|ies)|acknowledg(?:e)?ments?)\b", title, re.IGNORECASE))


def _parse_source(source_html: str) -> tuple[str, list[dict], dict]:
    parser = _TreeParser()
    parser.feed(source_html)
    root = parser.root

    title = _first_text(root, {"h1"})
    abstract_node = next((node for node in _walk(root) if _has_class(node, "ltx_abstract")), None)
    if abstract_node is None:
        raise ValueError("arXiv HTML did not contain an abstract")
    abstract_parts = [_text(node) for node in _descendants(abstract_node, tag="p")]
    abstract = "\n\n".join(part for part in abstract_parts if part)
    if not abstract:
        raise ValueError("arXiv HTML abstract was empty")

    elements: list[dict] = [
        {
            "element_id": "text_abstract",
            "type": "text",
            "section": "Abstract",
            "page": None,
            "content": abstract,
            "caption": "",
            "mentions": [],
            "image_path": None,
            "source_url": SOURCE_URL,
            "source_location": "Abstract",
        }
    ]

    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for paragraph in _descendants(root, tag="p", class_name="ltx_p"):
        if _inside(paragraph, lambda n: _has_class(n, "ltx_abstract") or _has_class(n, "ltx_table")):
            continue
        section = _nearest_section(paragraph)
        section_title = _section_title(section)
        if not section_title or _is_references(section_title):
            continue
        text = _text(paragraph)
        if text:
            grouped.setdefault(section_title, []).append(text)

    for title_name, paragraphs in grouped.items():
        slug = re.sub(r"[^a-z0-9]+", "_", title_name.lower()).strip("_") or "section"
        elements.append(
            {
                "element_id": f"text_{slug}",
                "type": "text",
                "section": title_name,
                "page": None,
                "content": "\n\n".join(paragraphs),
                "caption": "",
                "mentions": [],
                "image_path": None,
                "source_url": SOURCE_URL,
                "source_location": f"Section: {title_name}",
            }
        )

    table_figure = next(
        (
            node
            for node in _walk(root)
            if _has_class(node, "ltx_table")
            and re.search(r"\bTable\s*4\b", _first_text(node, {"figcaption"}), re.IGNORECASE)
        ),
        None,
    )
    if table_figure is None:
        raise ValueError("arXiv HTML did not contain Table 4")

    caption = _first_text(table_figure, {"figcaption"})
    table = next(iter(_descendants(table_figure, tag="table")), None)
    if table is None:
        raise ValueError("Table 4 did not contain an HTML table")
    rows: list[list[str]] = []
    for row in _descendants(table, tag="tr"):
        cells = []
        for cell in _descendants(row):
            if cell.tag in {"td", "th"} and not _inside(cell, lambda n: n is not row and n.tag == "tr"):
                value = _clean_table_cell(_text(cell))
                if value:
                    cells.append(value)
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        raise ValueError(f"Table 4 yielded too few rows: {rows!r}")

    headers = rows[0]
    data_rows = rows[1:]
    if not any(row and row[0] == "KEHNN" for row in data_rows):
        raise ValueError("Table 4 source parse did not yield the KEHNN row")
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join("---" for _ in headers) + " |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in data_rows)
    table_data = {"headers": headers, "rows": data_rows, "caption": caption}
    table_update = {
        "element_id": "elem_004",
        "type": "table",
        "content": markdown,
        "caption": caption,
        "table_data": table_data,
        "source_url": SOURCE_URL,
        "source_location": "Experiments > Response Selection > Table 4",
    }
    return title, elements, table_update


def _fetch_source(url: str, path: Path, *, refresh: bool) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        request = Request(url, headers={"User-Agent": "phase9h-paper-restore/1.0"})
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed documented arXiv URL
            data = response.read()
        path.write_bytes(data)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8"), digest


def restore_target_paper(*, refresh: bool = False, source_url: str = SOURCE_URL) -> dict:
    """Fetch, parse, and merge the target paper into processed_papers.json."""
    source_path = SOURCE_HTML_PATH if source_url == SOURCE_URL else SOURCE_DIR / f"{PAPER_ID}.custom.html"
    source_html, digest = _fetch_source(source_url, source_path, refresh=refresh)
    title, text_elements, table_update = _parse_source(source_html)

    with PAPERS_PATH.open(encoding="utf-8") as handle:
        papers = json.load(handle)
    target = next((paper for paper in papers if paper.get("paper_id") == PAPER_ID), None)
    if target is None:
        raise ValueError(f"{PAPER_ID} was not found in {PAPERS_PATH}")

    existing = [dict(element) for element in target.get("elements", [])]
    existing = [element for element in existing if not str(element.get("element_id", "")).startswith("text_")]
    table_found = False
    for element in existing:
        if element.get("element_id") == table_update["element_id"]:
            element.update(table_update)
            table_found = True
            break
    if not table_found:
        raise ValueError("Existing SPIQA target did not contain elem_004 / Table 4")
    target["title"] = title
    target["elements"] = existing + text_elements

    with PAPERS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(papers, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    SOURCE_META_PATH.write_text(
        json.dumps(
            {
                "paper_id": PAPER_ID,
                "source": "arXiv",
                "source_version": PAPER_ID,
                "source_url": source_url,
                "sha256": digest,
                "restored_elements": [element["element_id"] for element in text_elements] + ["elem_004"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "paper_id": PAPER_ID,
        "title": title,
        "source_url": source_url,
        "sha256": digest,
        "text_elements": len(text_elements),
        "table_headers": table_update["table_data"]["headers"],
        "table_rows": len(table_update["table_data"]["rows"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch the versioned arXiv HTML")
    parser.add_argument("--source-url", default=SOURCE_URL, help="versioned HTML source URL")
    args = parser.parse_args()
    result = restore_target_paper(refresh=args.refresh, source_url=args.source_url)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
