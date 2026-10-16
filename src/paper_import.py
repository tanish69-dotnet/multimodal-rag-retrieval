"""Online paper import (Phase 9P+: arXiv search → PDF → existing ingest path).

Flow: ``GET /papers/search?q=...`` (arXiv Atom API, read-only metadata) →
user picks an entry → ``POST /papers/import`` downloads the PDF from an
allowlisted arXiv host (export.arxiv.org / arxiv.org, following redirects
only within those hosts) → validates magic/size → reuses
``parse_pdf_bytes`` + ``ingest_paper`` (paper-scoped replace, never global
reset). No credentials, no paywalls, no unrestricted URL fetch: only
``https://`` URLs on the arXiv allowlist, size/timeout/MIME enforced,
redirects re-validated hop by hop.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

ALLOWED_HOSTS = {"export.arxiv.org", "arxiv.org"}
ALLOWED_SCHEMES = {"https"}
SEARCH_URL = "https://export.arxiv.org/api/query"
PDF_URL_TMPL = "https://arxiv.org/pdf/{arxiv_id}"
SEARCH_TIMEOUT_S = 20.0
DOWNLOAD_TIMEOUT_S = 120.0
MAX_PDF_BYTES = 32 * 1024 * 1024  # same cap as /ingest-pdf

_ARXIV_ID_RE = re.compile(r"\b(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    pdf_url: str
    source_url: str


def extract_arxiv_id(text: str) -> str | None:
    match = _ARXIV_ID_RE.search(text or "")
    return match.group(1) if match else None


def _is_allowed_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    try:
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme in ALLOWED_SCHEMES
        and (parts.hostname or "").lower() in ALLOWED_HOSTS
        and parts.username is None
        and parts.password is None
        and port in (None, 443)
    )


def search_arxiv(query: str, *, max_results: int = 5) -> list[PaperCandidate]:
    """Search arXiv metadata (no PDF download). Raises on transport errors."""
    import httpx

    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, min(int(max_results or 5), 10)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    with httpx.Client(timeout=SEARCH_TIMEOUT_S, follow_redirects=False) as client:
        response = client.get(SEARCH_URL, params=params)
        response.raise_for_status()
        feed = response.text
    root = ET.fromstring(feed)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out: list[PaperCandidate] = []
    for entry in root.findall("atom:entry", ns):
        id_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        arxiv_id = extract_arxiv_id(id_url) or ""
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf" and (link.get("href") or "").strip():
                pdf_url = link.get("href").strip()
                break
        if not pdf_url and arxiv_id:
            pdf_url = PDF_URL_TMPL.format(arxiv_id=arxiv_id)
        if not arxiv_id or not _is_allowed_url(pdf_url):
            continue
        out.append(
            PaperCandidate(
                arxiv_id=arxiv_id,
                title=title,
                authors=[a for a in authors if a],
                summary=summary,
                published=published,
                pdf_url=pdf_url,
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return out


def download_pdf_bytes(pdf_url: str) -> bytes:
    """Download a PDF from an allowlisted arXiv host with strict validation."""
    import httpx  # noqa: F811 - local import keeps module importable without httpx

    if not _is_allowed_url(pdf_url):
        raise ValueError(f"URL host not allowed: {pdf_url!r}")
    with httpx.Client(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=False) as client:
        url, hops, data = pdf_url, 0, b""
        while True:
            if hops > 3:
                raise ValueError("too many redirects")
            # Stream production responses so a hostile or broken server cannot
            # force an oversized PDF into memory before the cap is checked.
            if hasattr(client, "stream"):
                with client.stream("GET", url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location", "")
                        nxt = urllib.parse.urljoin(url, location)
                        if not _is_allowed_url(nxt):
                            raise ValueError(f"redirect target not allowed: {nxt!r}")
                        url, hops = nxt, hops + 1
                        continue
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").lower()
                    if "pdf" not in content_type and "octet-stream" not in content_type:
                        raise ValueError(f"unexpected content-type {content_type!r} (expected PDF)")
                    declared_length = response.headers.get("content-length")
                    if declared_length and int(declared_length) > MAX_PDF_BYTES:
                        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                    buf = bytearray()
                    for chunk in response.iter_bytes(1024 * 1024):
                        buf.extend(chunk)
                        if len(buf) > MAX_PDF_BYTES:
                            raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                    data = bytes(buf)
                    break

            # Compatibility path for small test doubles that implement only
            # ``get``; real httpx clients always take the streaming branch.
            response = client.get(url)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location", "")
                nxt = urllib.parse.urljoin(url, location)
                if not _is_allowed_url(nxt):
                    raise ValueError(f"redirect target not allowed: {nxt!r}")
                url, hops = nxt, hops + 1
                continue
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "pdf" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"unexpected content-type {content_type!r} (expected PDF)")
            data = response.content
            if len(data) > MAX_PDF_BYTES:
                raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
            break
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
    if data[:5] != b"%PDF-":
        raise ValueError("downloaded file is not a PDF (missing %PDF- magic)")
    return data


def import_arxiv_paper(
    arxiv_id: str,
    service: Any,
    *,
    title_override: str | None = None,
    authors: list[str] | None = None,
    abstract: str = "",
    source_url: str | None = None,
) -> dict[str, Any]:
    """Search → download → parse → index one arXiv paper (existing pipeline).

    Returns the same status shape as ``POST /ingest-pdf`` plus ``source_url``
    and ``download_bytes``. Never clears other papers (service does
    paper-scoped replace).
    """
    clean = extract_arxiv_id(arxiv_id or "")
    if not clean:
        raise ValueError(f"not a valid arXiv identifier: {arxiv_id!r}")
    pdf_url = PDF_URL_TMPL.format(arxiv_id=clean)
    started = time.perf_counter()
    raw = download_pdf_bytes(pdf_url)
    from src.ingest.pdf_ingest import parse_pdf_bytes

    paper, warnings = parse_pdf_bytes(raw, paper_id=clean, title=title_override)
    if authors:
        paper["authors"] = [author.strip() for author in authors if author.strip()]
    if abstract.strip():
        paper["abstract"] = abstract.strip()
    paper["source_url"] = source_url or f"https://arxiv.org/abs/{clean}"
    chunks = service.ingest_paper(paper)
    elements = paper.get("elements", [])
    return {
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "source_url": paper["source_url"],
        "download_bytes": len(raw),
        "elements": len(elements),
        "text_elements": sum(1 for e in elements if e.get("type") == "text"),
        "table_elements": sum(1 for e in elements if e.get("type") == "table"),
        "figure_elements": sum(1 for e in elements if e.get("type") == "figure"),
        "figures_with_bytes": sum(
            1 for e in elements if e.get("type") == "figure" and e.get("image_path")
        ),
        "chunks_created": chunks,
        "warnings": warnings,
        "latency_s": round(time.perf_counter() - started, 1),
    }
