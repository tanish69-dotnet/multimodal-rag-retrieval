"""Inspect the SPIQA dataset (google/spiqa) structure.

Streams records from Hugging Face and prints:
  - the dataset feature schema (top-level keys, types)
  - a health check over the stream: unique papers, ground-truth reference
    accessibility (reference_figure present in all_figures?), caption metadata
    availability, and content-type distribution
  - N paper records showing question, answer, and ground-truth figure/table
    targets with their captions

Schema discovered for the HF `test` split (666 rows, QA-pair-per-row):
  paper_id         : str, arXiv id of the paper (e.g. "1611.04684v1")
  question         : str
  answer           : str
  rationale        : str
  reference_figure : str, ground-truth target PNG name (e.g. "1611.04684v1-Table4-1.png")
  all_figures      : dict {figure_name: {caption, content_type, figure_type}}

Run from the project root:  python src/inspect_spiqa.py [--samples 5] [--limit N]
"""

import argparse
import itertools
import json
from collections import Counter, defaultdict

from datasets import load_dataset

HF_NAME = "google/spiqa"
DEFAULT_SAMPLES = 5
TRUNCATE = 180


def truncate(value: object, width: int = TRUNCATE) -> str:
    """Render a value as text, truncated to a readable width."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 3] + "..."


def summarize_value(value, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, str):
        return f"{pad}str ({len(value)} chars): {truncate(value, 120)}"
    if isinstance(value, dict):
        if not value:
            return f"{pad}dict (empty)"
        head_key, head_val = next(iter(value.items()))
        head_kind = "dict" if isinstance(head_val, dict) else f"{type(head_val).__name__}"
        return (
            f"{pad}dict[{len(value)} keys], values are {head_kind}; "
            f"e.g. key={head_key!r} value={truncate(head_val, 90)}"
        )
    if isinstance(value, list):
        if not value:
            return f"{pad}list[] (empty)"
        return f"{pad}list[{len(value)}] of {type(value[0]).__name__}; e.g. {truncate(value[0])}"
    return f"{pad}{type(value).__name__}: {truncate(str(value), 120)}"


def content_type_from_name(name: str) -> str:
    """Infer element type from a SPIQA png name."""
    lowered = name.lower()
    if "table" in lowered:
        return "table"
    if "figure" in lowered:
        return "figure"
    return "unknown"


def meta_of(value) -> dict:
    """Normalize an `all_figures` value to a metadata dict (may be empty)."""
    return value if isinstance(value, dict) else {}


def print_features(features) -> None:
    print("=" * 78)
    print("FEATURE SCHEMA (top-level fields)")
    print("=" * 78)
    for name, feat in features.items():
        print(f"- {name}: {feat}")


def run_health_check(stream, limit: int | None) -> None:
    print("\n" + "=" * 78)
    print("SCHEMA HEALTH CHECK (streamed rows)")
    print("=" * 78)
    rows = itertools.islice(stream, None if limit is None else limit)

    total_rows = 0
    unique_papers = set()
    ref_in_figures = 0
    ref_meta_present = 0
    rows_with_own_figures = 0
    type_counter = Counter()
    figure_type_counter = Counter()
    sample_refs = []

    for sample in rows:
        total_rows += 1
        pid = sample.get("paper_id", "?")
        unique_papers.add(pid)
        ref = sample.get("reference_figure")
        figures = sample.get("all_figures") or {}

        if not isinstance(figures, dict):
            figures = dict(figures) if figures else {}

        own_figures = {k: v for k, v in figures.items() if str(k).startswith(str(pid) + "-")}
        if own_figures:
            rows_with_own_figures += 1
            for name, meta in own_figures.items():
                meta = meta_of(meta)
                type_counter[meta.get("content_type") or content_type_from_name(str(name))] += 1
                ftype = meta.get("figure_type")
                if ftype and ftype != "N/A":
                    figure_type_counter[ftype] += 1

        if ref:
            sample_refs.append(ref)
            type_counter["reference:" + (content_type_from_name(ref))] += 1
            if ref in figures:
                ref_in_figures += 1
            m = meta_of(figures.get(ref))
            if m:
                ref_meta_present += 1

    print(f"rows streamed                : {total_rows}")
    print(f"unique papers                : {len(unique_papers)} (>= {min(118, len(unique_papers))} expected on test split)")
    print(f"rows with own figures in dict: {rows_with_own_figures} / {total_rows}")
    print(f"reference target present in all_figures        : {ref_in_figures} / {total_rows}")
    print(f"reference target has caption metadata          : {ref_meta_present} / {total_rows}")
    print("content-type distribution (per row all_figures own keys + reference):")
    for label, count in sorted(type_counter.items()):
        print(f"  {label:20s}: {count}")
    if figure_type_counter:
        print("figure_type distribution:")
        for label, count in sorted(figure_type_counter.items()):
            print(f"  {label:20s}: {count}")
    print("sample reference (ground-truth) filenames:", ", ".join(sample_refs[:6]))


def print_paper(sample: dict, index: int) -> None:
    pid = sample.get("paper_id", "?")
    raw_figures = sample.get("all_figures") or {}
    figures = {k: meta_of(v) for k, v in raw_figures.items()} if isinstance(raw_figures, dict) else {}
    ref = sample.get("reference_figure")

    print("\n" + "=" * 78)
    print(f"PAPER #{index + 1}: {pid}")
    print("=" * 78)

    print("\nTop-level keys and structure:")
    for key, value in sample.items():
        print(f"  [{key}] {summarize_value(value, indent=6).lstrip()}")
        # summarize_value already prints an indented block; print plainly here.

    m = figures.get(ref, {}) if ref else {}
    ref_type = m.get("content_type") if m else content_type_from_name(ref or "")
    print(f"\nGround-truth target: {ref!r} -> type={ref_type or 'unknown'}")
    if m:
        print(f"  caption     : {m.get('caption', '')}")
        print(f"  content_type: {m.get('content_type')} | figure_type: {m.get('figure_type')}")
    else:
        print("  (no caption metadata for this target in this row's all_figures)")

    print("paper's figures/tables in all_figures:")
    if figures:
        for name, meta in figures.items():
            extra = ""
            if meta:
                extra = f" | caption={truncate(meta.get('caption', ''), 60)} | figure_type={meta.get('figure_type')}"
            print(f"  - {name} [{meta.get('content_type') or content_type_from_name(name)}]{extra}")
    else:
        print("  (none)")

    print(f"\nquestion : {sample.get('question')}")
    print(f"answer   : {truncate(sample.get('answer', ''), 260)}")
    print(f"rationale: {truncate(sample.get('rationale', ''), 260)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the SPIQA dataset schema")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="paper records to print in detail")
    parser.add_argument("--limit", type=int, default=None, help="cap on rows streamed for the health check")
    args = parser.parse_args()

    print(f"Loading {HF_NAME} (streaming)...\n")
    raw = load_dataset(HF_NAME, streaming=True)

    if isinstance(raw, dict):
        print(f"Available splits: {list(raw.keys())}")
        stream = raw[list(raw.keys())[0]]
    else:
        print("Dataset has no named splits (single stream).")
        stream = raw

    try:
        features = stream.features
        print_features(features)
    except Exception as exc:  # features not exposed on some iterables
        print(f"(Could not read feature schema from stream: {exc})")

    run_health_check(stream, args.limit)

    # A fresh stream is required for the detailed sample pass.
    raw2 = load_dataset(HF_NAME, streaming=True)
    stream2 = raw2[list(raw2.keys())[0]] if isinstance(raw2, dict) else raw2

    print(f"\nDetailed view of the first {args.samples} paper records:")
    for i, sample in enumerate(itertools.islice(stream2, args.samples)):
        print_paper(sample, i)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())