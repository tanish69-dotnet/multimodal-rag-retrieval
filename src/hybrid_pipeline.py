"""Hybrid retrieval: RRF fusion of dense (ChromaDB) + sparse (BM25), then
cross-encoder reranking (BAAI/bge-reranker-base).

Pipeline:
  hp = HybridPipeline()
  hits = hp.search(query, k=5)  # returns EvidenceItem schema objects

RRF: RRF(d) = sum (1 / (k_rrf + rank_i)) for systems i, with k_rrf=60.
Fuse top-15 from dense, top-15 from BM25 -> dedupe -> keep top-20 for rerank.
CrossEncoder scores (query, document_text) where document_text is the same
content string used for dense indexing; rerank by CE score desc to final k.

Extras:
  - Query Router: intent detection (table/figure/text) + score boosting.
  - Cross-Reference Expander: resolves "Table X" / "Figure Y" citations in text.
  - Per-paper filtering API: retrieve_for_papers for comparison matrices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

from sentence_transformers import CrossEncoder

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.query_router import classify_intent, boost_matching_type, Intent
from src.cross_ref_expander import expand_cross_references, ExpanderConfig

K_RRF = 60
DENSE_TOP = 15
SPARSE_TOP = 15
FUSE_TOP = 20


class HybridPipeline:
    def __init__(
        self,
        *,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        ce_model: str = "BAAI/bge-reranker-base",
        paper_elements_lookup: Optional[dict[str, list[dict]]] = None,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.ce = CrossEncoder(ce_model, trust_remote_code=False)
        # paper_id -> list of all elements (for cross-ref expansion)
        self._paper_elements = paper_elements_lookup or {}

    def _build_element_lookup(self, papers_data: Sequence[dict]) -> None:
        """Build paper_id -> elements map from paper records."""
        self._paper_elements = {}
        for p in papers_data:
            pid = p.get("paper_id")
            if pid:
                self._paper_elements[pid] = p.get("elements", [])

    def index_papers(self, papers_data: Sequence[dict], *, reset: bool = True) -> tuple[int, int]:
        if reset:
            self._build_element_lookup(papers_data)
        else:
            for paper in papers_data:
                pid = paper.get("paper_id")
                if pid:
                    self._paper_elements[pid] = paper.get("elements", [])
        nd = self.dense.index_papers(papers_data, reset=reset)
        ns = self.sparse.index_papers(papers_data, reset=reset)
        return nd, ns

    def delete_paper(self, paper_id: str) -> tuple[int, int]:
        """Remove one paper's chunks from both indexes (recoverable replace).

        Never touches other papers and never clears the global index. Returns
        ``(dense_removed, sparse_removed)``. The element lookup is dropped for
        the paper so cross-ref expansion cannot serve stale elements.
        """
        dense_removed = self.dense.delete_paper(paper_id)
        sparse_removed = self.sparse.delete_paper(paper_id)
        self._paper_elements.pop(paper_id, None)
        return dense_removed, sparse_removed

    def _base_search(self, query: str, paper_ids: Optional[Sequence[str]] = None) -> tuple[list[dict], list[dict]]:
        """Run dense + BM25, return raw hits."""
        dens = self.dense.retrieve(query, paper_ids=paper_ids, k=DENSE_TOP)
        spar = self.sparse.retrieve(query, paper_ids=paper_ids, k=SPARSE_TOP)
        return dens, spar

    def search(
        self,
        query: str,
        k: int = 5,
        paper_ids: Optional[Sequence[str]] = None,
        *,
        use_router: bool = True,
        use_expander: bool = True,
    ) -> tuple[list[dict], dict]:
        """Full pipeline with optional query routing and cross-ref expansion.

        Args:
            query: user query string.
            k: final number of results after reranking.
            paper_ids: optional list of paper_ids to restrict search.
            use_router: if True, apply intent detection + type boosting.
            use_expander: if True, expand cross-references from top text hits.

        Returns:
            (reranked_hits, timing_stats)
        """
        t0 = time.perf_counter()
        dens, spar = self._base_search(query, paper_ids)
        t_dense = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused = self._rrf_fuse(dens, spar)
        t_fuse = time.perf_counter() - t0

        # Query Router: boost matching element types
        if use_router:
            intent = classify_intent(query)
            if intent != "general":
                boost_matching_type(fused, intent)
                fused.sort(key=lambda h: h.get("score", 0), reverse=True)
        else:
            intent = "general"

        # Cross-Reference Expansion: resolve Table/Figure refs in top text hits
        expanded = fused
        if use_expander:
            expanded = expand_cross_references(
                fused,
                get_paper_elements=lambda pid: self._paper_elements.get(pid, []),
                config=ExpanderConfig(max_expansions=4, append=True, deduplicate=True),
            )

        t0 = time.perf_counter()
        reranked = self._rerank(query, expanded, k, intent=intent)
        t_rerank = time.perf_counter() - t0
        t_total = t_dense + t_fuse + t_rerank

        for i, h in enumerate(reranked, 1):
            h["rank"] = i
        return reranked, {
            "t_dense": t_dense,
            "t_fuse": t_fuse,
            "t_rerank": t_rerank,
            "t_total": t_total,
            "n_dense": len(dens),
            "n_sparse": len(spar),
            "n_fused": len(fused),
            "n_expanded": len(expanded),
            "intent": intent,
        }

    def retrieve_for_papers(
        self,
        query: str,
        paper_ids: Sequence[str],
        k_per_paper: int = 3,
        *,
        use_router: bool = True,
        use_expander: bool = True,
    ) -> dict[str, list[dict]]:
        """Retrieve top-k evidence per paper for cross-paper comparison (Person 4).

        Runs independent search per paper, applies router + expander per paper,
        and returns a dict {paper_id: [hits...]} with exactly k_per_paper items
        (or fewer if corpus lacks enough). Useful for side-by-side comparison
        tables across papers.

        Args:
            query: question / search text.
            paper_ids: list of paper_ids to query.
            k_per_paper: max results per paper.
            use_router: apply intent detection + boosting per paper.
            use_expander: apply cross-ref expansion per paper.

        Returns:
            dict mapping paper_id -> list of EvidenceItem dicts (rank 1..k).
        """
        out: dict[str, list[dict]] = {}
        for pid in paper_ids:
            hits, _ = self.search(
                query,
                k=k_per_paper,
                paper_ids=[pid],
                use_router=use_router,
                use_expander=use_expander,
            )
            # ensure local ranking
            for i, h in enumerate(hits, 1):
                h["rank"] = i
            out[pid] = hits
        return out

    def _key(self, h: dict) -> str:
        pid = h.get("paper_id") or ""
        eid = h.get("element_id") or ""
        cid = h.get("chunk_id") or ""
        return f"{pid}::{eid}::{cid}" if cid else f"{pid}::{eid}"

    def _rrf_fuse(self, dens: list[dict], spar: list[dict]) -> list[dict]:
        rrf = {}
        # dense: rank starts at 1
        for d in dens:
            k = self._key(d)
            rank = int(d.get("rank", getattr(d, "rank", 0)) or len(dens))
            sc = 1.0 / (K_RRF + rank)
            base = dict(d)
            rrf[k] = {"score": sc, "item": base, "from": {**base, "score_dense": base.get("score"), "rrf": sc}}
        for s in spar:
            k = self._key(s)
            rank = int(s.get("rank_bm25", 0) or len(spar))
            sc = 1.0 / (K_RRF + rank)
            if k in rrf:
                rrf[k]["score"] += sc
                rrf[k]["item"].setdefault("score_sparse", s.get("bm25_score"))
                rrf[k]["item"].setdefault("rank_bm25", rank)
            else:
                base = dict(s)
                base["score_sparse"] = s.get("bm25_score")
                rrf[k] = {"score": sc, "item": base, "from": base}
        items = list(rrf.values())
        items.sort(key=lambda x: x["score"], reverse=True)
        return [x["item"] for x in items[:FUSE_TOP]]

    def _rerank(self, query: str, fused: list[dict], k: int, *, intent: Intent = "general") -> list[dict]:
        if not fused:
            return []
        texts = [f["content"] or "" for f in fused]
        try:
            ce_scores = self.ce.predict([(query, t) for t in texts])
        except Exception:
            ce_scores = [0.0] * len(texts)
        target_type = "table" if intent == "table" else "figure" if intent == "figure" else "text"
        pairs = []
        for i, h in enumerate(fused):
            cs = float(ce_scores[i]) if i < len(ce_scores) else 0.0
            hh = dict(h)
            hh["ce_score"] = cs
            # The router used to boost RRF scores before reranking, after
            # which the cross-encoder discarded the boost entirely. Apply the
            # same small type-aware multiplier to the final ranking score,
            # while preserving the raw CE score for observability.
            hh["intent_score"] = cs * 1.15 if intent != "general" and hh.get("type") == target_type else cs
            pairs.append(hh)
        pairs.sort(key=lambda x: x["intent_score"], reverse=True)
        out = pairs[:k]
        for i, h in enumerate(out, 1):
            h["rank"] = i
        return out
