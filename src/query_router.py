"""Query Router: rule-based intent classification + score boosting.

Given a user query, classifies intent into ``table``, ``figure``, ``text``,
or ``general`` and provides a boosting function that increases the score of
candidate elements whose ``type`` matches the detected intent.
"""

from __future__ import annotations

import re
from typing import Callable, Literal

Intent = Literal["table", "figure", "text", "general"]

# Intent -> keywords (case-insensitive).
INTENT_KEYWORDS: dict[Intent, list[str]] = {
    "table": [
        "table", "benchmark", "accuracy", "bleu", "score", "metric",
        "ablation", "performance", "results", "statistics", "comparison",
        "f1", "precision", "recall", "auc", "perplexity", "rouge",
        "best", "highest", "lowest", "worst", "difference", "delta",
        "rank", "row", "column", "reported",
    ],
    "figure": [
        "figure", "fig", "plot", "chart", "diagram", "curve",
        "architecture", "overview", "visualization", "visualise", "visualize",
        "schematic", "illustration", "graph", "heatmap", "tsne", "umap",
    ],
    "text": [
        "method", "methodology", "approach", "propose", "architecture",
        "model", "training", "loss", "optimizer", "dataset", "framework",
        "algorithm", "theory", "analysis", "proof", "concept", "design",
        "explain", "describe", "mechanism",
    ],
}

_TABLE_INTENT_RE = re.compile(
    r"\b(?:"
    r"which\s+(?:method|model|system|approach|variant)\s+(?:perform\w*|achieve\w*|has|have|is)\s+(?:the\s+)?(?:best|highest|lowest|top|worst)"
    r"|(?:best|highest|lowest|top|worst)\s+(?:method|model|system|approach|variant|value|score|result)"
    r"|compar(?:e|ison)\b|difference\s+between\b|summari[sz]e\s+(?:the\s+)?(?:table|experimental|reported)?\s*results?"
    r")\b",
    re.IGNORECASE,
)
_FIGURE_REFERENCE_RE = re.compile(r"\b(?:figure|fig\.?)\s*\d", re.IGNORECASE)
_TABLE_REFERENCE_RE = re.compile(r"\btable\s*\d", re.IGNORECASE)


def classify_intent(query: str) -> Intent:
    """Classify query intent by keyword overlap (simple, deterministic).

    Returns ``general`` if no strong signal.
    """
    q = query.lower()
    # Explicit references and focused comparative wording are more reliable
    # than generic words such as "model" or "performance".
    if _TABLE_REFERENCE_RE.search(q):
        return "table"
    if _FIGURE_REFERENCE_RE.search(q):
        return "figure"
    if _TABLE_INTENT_RE.search(q):
        return "table"
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", q))
        scores[intent] = hits
    if not any(scores.values()):
        return "general"
    # prefer table > figure > text on tie (table/figure queries usually more specific)
    best = max(scores, key=lambda k: (scores[k], -list(INTENT_KEYWORDS).index(k)))
    return best


def boost_matching_type(
    hits: list[dict],
    intent: Intent,
    *,
    additive_boost: float = 0.15,
    mult_boost: float = 1.15,
) -> list[dict]:
    """Boost scores of elements whose type matches the detected intent.

    - For RRF scores: additive_boost (e.g., +0.15).
    - For cross-encoder / dense scores: multiplicative_boost (e.g., ×1.15).
    Modifies ``hits`` in place and returns it for chaining.

    Args:
        hits: list of candidate dicts (EvidenceItem-like, with ``type``).
        intent: detected intent from ``classify_intent``.
        additive_boost: added to ``score`` / ``rrf_score`` / ``bm25_score``.
        mult_boost: multiplies ``score`` / ``ce_score`` / dense similarity.

    Returns:
        Same list (mutated) with boosted scores and ``intent_boosted`` flag.
    """
    if intent == "general":
        return hits

    target_type = "table" if intent == "table" else "figure" if intent == "figure" else "text"
    for h in hits:
        h_type = h.get("type")
        if h_type == target_type:
            # boost dense/semantic score
            for key in ("score", "ce_score", "bm25_score", "rrf_score"):
                if key in h and h[key] is not None:
                    if key in ("ce_score", "score"):  # similarity-like [0,1]
                        h[key] = float(h[key]) * mult_boost
                    else:
                        h[key] = float(h[key]) + additive_boost
            h["intent_boosted"] = True
            h["matched_intent"] = intent
    return hits


def search_with_routing(
    base_retrieve: Callable[[str], list[dict]],
    query: str,
    *,
    k: int = 10,
    boost: bool = True,
) -> tuple[list[dict], Intent]:
    """Convenience: classify intent, retrieve, optionally boost, and return."""
    intent = classify_intent(query)
    hits = base_retrieve(query, k=k)
    if boost and intent != "general":
        boost_matching_type(hits, intent)
        hits.sort(key=lambda h: h.get("score", h.get("ce_score", 0)), reverse=True)
    return hits, intent
