"""Cross-Reference Expander: resolves in-text citations (e.g., "Table 1", "Figure 2")
to their corresponding elements in the same paper.

Given a set of retrieved text chunks, scans for patterns like
  "Table 1", "Tab. 2", "Figure 3", "Fig. 4", "Table 1a", "Figure 2b"
and fetches the matching table/figure element from the same paper by
normalising the reference label against stored element metadata.

Returns an augmented list of hits with extra entries for the resolved
references, flagged with ``expanded_via_cross_ref=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

# Matches: Table 1, Tab. 1, Figure 2, Fig. 2, Table 1a, Figure 2b, etc.
_REF_PATTERN = re.compile(
    r"\b(?:Table|Tab\.|Figure|Fig\.)\s*([0-9][0-9A-Za-z\-]*)\b",
    flags=re.IGNORECASE,
)


@dataclass
class ExpanderConfig:
    """Configuration for the expander behavior."""
    # Maximum number of cross-ref elements to append per query.
    max_expansions: int = 4
    # Whether to append expansions to the END of the ranked list or
    # insert them near the referring text chunk.
    append: bool = True
    # If True, duplicates by element_id are deduplicated.
    deduplicate: bool = True


def _normalize_label(label: str) -> str:
    """Normalize a table/figure label to a key suitable for matching.

    Removes trailing letters (e.g., "1a" -> "1") and lowercases.
    """
    m = re.match(r"([0-9]+)", label)
    return m.group(1) if m else label.lower()


def _element_key(element: dict) -> str:
    """Build a matching key from an element's caption or source_image.

    Uses the numeric part of the caption (e.g., "Table 1: ...") or
    the source_image filename ("...-Table1-1.png").
    """
    for field in ("caption", "source_image"):
        val = element.get(field) or ""
        if not val:
            continue
        # find first "Table N" or "Figure N" in caption
        m = re.search(r"(?:Table|Figure)\s*([0-9]+)", val, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)}"
        # try source_image pattern "arxiv-Table1-1.png"
        m = re.search(r"(?:Table|Figure)([0-9]+)", val, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def _find_matching_elements(
    paper_elements: Iterable[dict],
    ref_label: str,
) -> list[dict]:
    """Return elements from the same paper whose caption/image matches the label."""
    norm = _normalize_label(ref_label)
    matches = []
    for e in paper_elements:
        ek = _element_key(e)
        if ek == norm:
            matches.append(e)
    return matches


def expand_cross_references(
    hits: list[dict],
    *,
    get_paper_elements: Callable[[str], list[dict]],
    config: Optional[ExpanderConfig] = None,
) -> list[dict]:
    """Expand cross-references found in top-ranked text hits.

    Args:
        hits: ranked retrieval results (EvidenceItem-like dicts).
        get_paper_elements: function(paper_id) -> list of all elements
                            (text, table, figure) for that paper.
        config: optional ExpanderConfig.

    Returns:
        New list of hits with cross-reference expansions appended/inserted.
    """
    config = config or ExpanderConfig()
    seen = set()
    expanded: list[dict] = []
    expansions_added = 0

    for hit in hits:
        # deduplicate by element_id if requested
        eid = hit.get("element_id")
        if config.deduplicate and eid:
            if eid in seen:
                continue
            seen.add(eid)

        expanded.append(hit)

        if expansions_added >= config.max_expansions:
            continue

        # Only text elements can have cross-references we care about
        if hit.get("type") != "text":
            continue

        pid = hit.get("paper_id")
        if not pid:
            continue

        # find all "Table X" / "Figure Y" mentions in the chunk text
        text = hit.get("content") or hit.get("context") or ""
        refs = _REF_PATTERN.findall(text)
        if not refs:
            continue

        # fetch all elements of this paper once
        paper_elements = list(get_paper_elements(pid))

        for ref in refs:
            matches = _find_matching_elements(paper_elements, ref)
            for m in matches:
                if config.deduplicate and m.get("element_id") in seen:
                    continue
                # build a hit-like dict from the element
                exp_hit = {
                    "paper_id": pid,
                    "element_id": m.get("element_id"),
                    "type": m.get("type"),
                    "page": m.get("page"),
                    "section": m.get("section"),
                    "content": m.get("content") or "",
                    "caption": m.get("caption"),
                    "context": " ".join(m.get("mentions") or []),
                    "image_path": m.get("image_path"),
                    "source_image": m.get("source_image"),
                    "figure_type": m.get("figure_type"),
                    "rank": None,
                    "score": None,
                    "ce_score": None,
                    "expanded_via_cross_ref": True,
                    "cross_ref_trigger": ref,
                }
                if config.append:
                    expanded.append(exp_hit)
                else:
                    # insert after the referring hit
                    expanded.insert(len(expanded), exp_hit)
                seen.add(m.get("element_id"))
                expansions_added += 1
                if expansions_added >= config.max_expansions:
                    break
        if expansions_added >= config.max_expansions:
            break

    return expanded