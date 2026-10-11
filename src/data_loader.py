"""Data loader: convert SPIQA (google/spiqa) records into our schema.md format.

Hugging Face `google/spiqa` (test split, 666 rows) uses a **QA-pair-per-row**
layout:

    paper_id         : str, arXiv id (e.g. "1611.04684v1")
    question / answer / rationale : str
    reference_figure : str, ground-truth target PNG (e.g. "1611.04684v1-Table4-1.png")
    all_figures      : dict {padded_png_name: {caption, content_type, figure_type}}

Mapping to schema.md
--------------------
The dataset provides no paper title, page/section offsets, or full text on this
split, and images ship separately (~GBs), so the conversion applies these rules:

  - `paper_id`   <- row["paper_id"]
  - `title`      <- row["title"] / row["paper_title"] if present, else "" (unavailable on split)
  - `elements`   <- one entry per figure/table in `all_figures` whose key belongs
                    to this paper (prefix `<paper_id>-`):
                        type               = content_type ("table" | "figure"),
                                              inferred from the filename as fallback
                        caption            = all_figures[key]["caption"]
                        section/page       = "" / None (not provided)
                        content            = "" (no cell/stroke extraction yet)
                        image_path         = None (images not downloaded)
                        source_image *ext* = the SPIQA PNG name (for later image
                                              retrieval / ground-truth linkage)
                        figure_type *ext*  = all_figures[key]["figure_type"]
  - QA ground truth: every QA record is linked to its ground-truth element via
    `reference_element_id`, resolved by matching `reference_figure` against the
    paper's elements (`source_image`).

Exported API
------------
  convert_spiqa_to_schema(sample)  -> schema dict for ONE row (paper + elements
                                      + its single question)
  load_working_subset(num_papers)  -> streams all rows once, groups by paper,
                                      converts the first `num_papers` papers,
                                      and saves:
                                          data/processed_papers.json   (papers)
                                          data/evaluation_questions.json (QAs)
                                      Returns (papers, questions).
  load_local_subset()              -> reads the saved subset back as
                                      (papers, questions).

Usage:
    python src/data_loader.py --num-papers 50            # build
    python src/data_loader.py --force                    # rebuild even if cached
    python src/data_loader.py --load-only                # read cached subset
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import load_dataset

HF_NAME = "google/spiqa"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PAPERS_PATH = DATA_DIR / "processed_papers.json"
QUESTIONS_PATH = DATA_DIR / "evaluation_questions.json"

DEFAULT_NUM_PAPERS = 50


# --------------------------------------------------------------------------- #
# Element helpers
# --------------------------------------------------------------------------- #
def _content_type(filename: str, meta: dict) -> str:
    """Resolve element type from metadata, falling back to the filename."""
    meta_type = meta.get("content_type")
    if meta_type in ("table", "figure"):
        return meta_type
    lowered = filename.lower()
    if "table" in lowered:
        return "table"
    if "figure" in lowered:
        return "figure"
    return "figure"


def _elements_from_figures(paper_id: str, figures) -> list[dict]:
    """Build schema `elements` from a row's `all_figures` dict.

    Only entries whose key starts with ``<paper_id>-`` belong to this paper.
    """
    if not isinstance(figures, dict):
        figures = {}
    own = {name: (meta if isinstance(meta, dict) else {}) for name, meta in figures.items()
           if str(name).startswith(str(paper_id) + "-")}

    elements = []
    for idx, (name, meta) in enumerate(sorted(own.items())):
        etype = "table" if _content_type(name, meta) == "table" else "figure"
        elements.append({
            "element_id": f"elem_{idx:03d}",
            "type": etype,
            "section": "",
            "page": None,
            "content": "",  # SPIQA does not expose table cells / text crops on this split
            "caption": meta.get("caption", "") or "",
            "mentions": [],
            "image_path": None,  # images ship separately; see `source_image`
            # --- auxiliary fields (beyond the schema.md minimum) ---
            "source_image": str(name),          # SPIQA png name, e.g. "1611.04684v1-Table4-1.png"
            "figure_type": meta.get("figure_type", "") or "",
        })
    return elements


def _make_question(sample: dict, elements: list[dict], q_index: int) -> dict:
    """Build one evaluation-question record with ground-truth element linkage."""
    paper_id = str(sample.get("paper_id", "") or "")
    ref = sample.get("reference_figure") or ""
    target = next((e for e in elements if e["source_image"] == ref), None)
    return {
        "question_id": f"{paper_id}__q{q_index:03d}",
        "paper_id": paper_id,
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "rationale": sample.get("rationale", ""),
        "reference_figure": ref or None,
        "reference_type": target["type"] if target else None,   # "table" | "figure"
        "reference_element_id": target["element_id"] if target else None,
        "reference_caption": target["caption"] if target else None,
    }


# --------------------------------------------------------------------------- #
# Public conversion API
# --------------------------------------------------------------------------- #
def convert_spiqa_to_schema(sample: dict) -> dict:
    """Convert one SPIQA row into our schema.md format.

    Args:
        sample: a single `google/spiqa` record (dict).

    Returns:
        dict with ``paper_id``, ``title``, ``elements`` (schema.md format), and
        ``questions`` (the row's single QA linked to its ground-truth element).
    """
    paper_id = str(sample.get("paper_id", "") or "")
    title = str(sample.get("paper_title") or sample.get("title") or "")
    elements = _elements_from_figures(paper_id, sample.get("all_figures"))
    questions = [_make_question(sample, elements, q_index=0)]
    return {
        "paper_id": paper_id,
        "title": title,
        "elements": elements,
        "questions": questions,
    }


# --------------------------------------------------------------------------- #
# Working subset
# --------------------------------------------------------------------------- #
def _aggregate_paper(rows: list[dict]) -> dict:
    """Merge all QA rows of one paper into a single schema paper record.

    Elements are deduplicated by image name (first non-empty metadata wins) and
    given stable ``elem_XXX`` ids sorted by image name, so ground-truth
    references resolve identically regardless of row order.
    """
    paper_id = str(rows[0].get("paper_id", "") or "")
    title = ""
    figures: dict[str, dict] = {}

    for row in rows:
        title = title or str(row.get("paper_title") or row.get("title") or "")
        raw = row.get("all_figures")
        if not isinstance(raw, dict):
            continue
        for name, meta in raw.items():
            meta = meta if isinstance(meta, dict) else {}
            name = str(name)
            if name.startswith(paper_id + "-") and (name not in figures or not figures[name]):
                figures[name] = {k: v for k, v in meta.items()}

    elements = _elements_from_figures(paper_id, figures)
    questions = [_make_question(row, elements, q_index) for q_index, row in enumerate(rows)]

    return {
        "paper_id": paper_id,
        "title": title,
        "elements": elements,
        "questions": questions,
    }


def _iter_rows() -> Iterator[dict]:
    """Yield every SPIQA row from the HF stream (groups by paper)."""
    raw = load_dataset(HF_NAME, streaming=True)
    stream = raw[list(raw.keys())[0]] if isinstance(raw, dict) else raw
    yield from stream


def load_working_subset(
    num_papers: int = DEFAULT_NUM_PAPERS,
    force: bool = False,
    data_dir: Optional[Path] = None,
) -> tuple[list, list]:
    """Build (or reuse) a local offline subset of SPIQA.

    Streams the dataset, groups rows by paper, converts the first ``num_papers``
    papers into schema.md format, and saves:

        <data_dir>/processed_papers.json      (list of paper schema records)
        <data_dir>/evaluation_questions.json  (list of QA ground-truth records)

    If both files already exist and ``force`` is False the cached files are
    reused, so later calls run fully offline.

    Returns:
        (papers, questions)
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    papers_path = data_dir / "processed_papers.json"
    questions_path = data_dir / "evaluation_questions.json"

    if papers_path.exists() and questions_path.exists() and not force:
        print(f"[cached] Reusing {papers_path} and {questions_path}. Use --force to rebuild.")
        return load_local_subset(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {HF_NAME} to build a {num_papers}-paper working subset...")

    # Group rows by paper, preserving first-seen order.
    papers_by_id: dict[str, list[dict]] = {}
    order: list[str] = []
    for sample in _iter_rows():
        pid = sample.get("paper_id")
        if pid not in papers_by_id:
            papers_by_id[pid] = []
            order.append(pid)
        papers_by_id[pid].append(sample)

    if not order:
        raise RuntimeError(f"Dataset {HF_NAME} yielded no records.")

    selected = order[:num_papers]
    unused = len(order) - len(selected)
    types = Counter()
    resolved = 0
    total_q = 0

    papers = []
    questions: list[dict] = []
    for pid in selected:
        paper = _aggregate_paper(papers_by_id[pid])
        papers.append({key: value for key, value in paper.items() if key != "questions"})
        for q in paper["questions"]:
            total_q += 1
            if q.get("reference_element_id"):
                resolved += 1
        for element in paper["elements"]:
            types[element["type"]] += 1
        questions.extend(paper["questions"])

    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(papers)} papers  -> {papers_path}")
    print(f"Saved {len(questions)} QAs  -> {questions_path}")
    print(f"  elements by type : {dict(types)}")
    print(f"  ground-truth links resolved: {resolved}/{total_q} questions")
    print(f"  papers left out of subset : {unused} (rest of the test split)")
    print("Note: figure/table image files are NOT bundled (they ship separately, ~GBs).")
    print("      `source_image` holds each SPIQA png name for later image discovery.")

    return papers, questions


def load_local_subset(data_dir: Optional[Path] = None) -> tuple[list, list]:
    """Read the locally saved working subset back as (papers, questions)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    with open(data_dir / "processed_papers.json", encoding="utf-8") as f:
        papers = json.load(f)
    with open(data_dir / "evaluation_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    return papers, questions


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SPIQA working subset for offline experiments")
    parser.add_argument("--num-papers", type=int, default=DEFAULT_NUM_PAPERS,
                        help=f"number of papers to include (default: {DEFAULT_NUM_PAPERS})")
    parser.add_argument("--force", action="store_true", help="rebuild even if the local subset exists")
    parser.add_argument("--load-only", action="store_true", help="print cached subset stats without touching the network")
    args = parser.parse_args()

    if args.load_only:
        papers, questions = load_local_subset()
    else:
        papers, questions = load_working_subset(num_papers=args.num_papers, force=args.force)

    n_elements = sum(len(p["elements"]) for p in papers)
    n_resolved = sum(1 for q in questions if q.get("reference_element_id"))
    print("\nSubset summary:")
    print(f"  papers={len(papers)}  questions={len(questions)}  elements={n_elements}")
    print(f"  QA records linked to a ground-truth element: {n_resolved}/{len(questions)}")
    if questions:
        print("  Example question:")
        q = questions[0]
        print(f"    {q['question_id']}: {q['question']}")
        print(f"    -> reference_element_id={q['reference_element_id']} ({q['reference_type']})")
        print(f"       reference_caption={q['reference_caption']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())