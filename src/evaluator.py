"""Evaluation utilities for SPIQA-style labeled questions.

Metrics computed (per query, then averaged):
- Recall@k / Hit@k: 1.0 if any ground-truth element appears in top-k results.
- MRR@10: 1 / rank of first relevant item, or 0.0 if not in top-10.
- Avg_Latency_sec: mean time to produce top-k results (seconds).

Relevance rule: a retrieved item is relevant if
  (item['element_id'] == gt['reference_element_id']) OR
  ((item['paper_id'], item['element_id']) matches ground truth).
GT is taken from ``data/evaluation_questions.json`` fields: ``reference_element_id``,
``reference_type``, ``paper_id``, ``reference_figure``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "evaluation_questions.json"


@dataclass
class QueryEval:
    question_id: str
    paper_id: Optional[str]
    gt_element_id: Optional[str]
    hits1: float
    hits5: float
    hits10: float
    mrr10: float
    latency: float
    top1_rel: bool


def _is_relevant(hit: dict, gt_elem: Optional[str], gt_pid: Optional[str]) -> bool:
    if gt_elem:
        if hit.get("element_id") == gt_elem:
            return True
    if gt_pid and gt_elem:
        if (hit.get("paper_id"), hit.get("element_id")) == (gt_pid, gt_elem):
            return True
    return False


def evaluate_questions(
    questions: list[dict],
    retrieve_fn: Callable[[str, Optional[list[str]]], tuple[list[dict], float]],
    *,
    k_max: int = 10,
    use_topk_for_latency: bool = True,
) -> dict[str, Any]:
    """Run retrieval_fn per question and compute averages.

    retrieve_fn(query, paper_ids=None) -> (hits, latency_sec) where hits are
    ranked results in order; latency_sec is time taken to produce those hits.
    """
    total = 0
    sum_h1 = sum_h5 = sum_h10 = sum_mrr = sum_lat = 0.0
    per_query: list[QueryEval] = []

    for q in questions:
        qid = q.get("question_id") or f"q{total}"
        pid = q.get("paper_id")
        gt_elem = q.get("reference_element_id")
        query = q.get("question") or ""

        t0 = time.perf_counter()
        hits, lat_fn = retrieve_fn(query, [pid] if pid else None)
        lat = lat_fn if isinstance(lat_fn, (int, float)) else (time.perf_counter() - t0)

        total += 1

        def rel_up_to(n):
            for h in hits[:n]:
                if _is_relevant(h, gt_elem, pid):
                    return True
            return False

        def first_rank_up_to(n):
            for i, h in enumerate(hits[:n], 1):
                if _is_relevant(h, gt_elem, pid):
                    return i
            return 0

        rank1 = first_rank_up_to(k_max)
        mrr10 = 1.0 / rank1 if rank1 > 0 else 0.0

        hits1 = 1.0 if rel_up_to(1) else 0.0
        hits5 = 1.0 if rel_up_to(5) else 0.0
        hits10 = 1.0 if rel_up_to(10) else 0.0

        sum_h1 += hits1
        sum_h5 += hits5
        sum_h10 += hits10
        sum_mrr += mrr10
        sum_lat += float(lat)

        per_query.append(
            QueryEval(
                question_id=qid,
                paper_id=pid,
                gt_element_id=gt_elem,
                hits1=hits1,
                hits5=hits5,
                hits10=hits10,
                mrr10=mrr10,
                latency=float(lat),
                top1_rel=rel_up_to(1),
            )
        )

    n = max(1, total)
    return {
        "n": n,
        "Recall@1": sum_h1 / n,
        "Recall@5": sum_h5 / n,
        "Recall@10": sum_h10 / n,
        "MRR@10": sum_mrr / n,
        "Avg_Latency_sec": sum_lat / n,
        "per_query": per_query,
    }


def save_csv(results_by_config: dict[str, dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Configuration",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "MRR@10",
                "Avg_Latency_sec",
            ]
        )
        for name in sorted(results_by_config):
            r = results_by_config[name]
            w.writerow(
                [
                    name,
                    f"{r['Recall@1']:.4f}",
                    f"{r['Recall@5']:.4f}",
                    f"{r['Recall@10']:.4f}",
                    f"{r['MRR@10']:.4f}",
                    f"{r['Avg_Latency_sec']:.4f}",
                ]
            )


def print_markdown_table(results_by_config: dict[str, dict[str, Any]]) -> None:
    cols = [
        "Configuration",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "Avg_Latency_sec",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for name in sorted(results_by_config):
        r = results_by_config[name]
        print(
            f"| {name} | {r['Recall@1']:.4f} | {r['Recall@5']:.4f} | "
            f"{r['Recall@10']:.4f} | {r['MRR@10']:.4f} | {r['Avg_Latency_sec']:.4f} |"
        )