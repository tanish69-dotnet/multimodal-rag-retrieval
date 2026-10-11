"""Unified Retrieval Service API for Person 2's RAG + Multimodal Retrieval pipeline.

This module provides a high-level `RetrievalService` class that encapsulates:
- HybridPipeline (dense + BM25 + RRF + cross-encoder rerank)
- QueryRouter (intent classification + score boosting)
- CrossRefExpander (in-text citation resolution)
- Per-paper filtering for Person 4's comparison matrices

All public methods return EvidenceItem dicts strictly conforming to `schema.md`:
    {rank, score, paper_id, element_id, type, page, section, content,
    caption, context, image_path, table_data, metadata}

Usage (programmatic):
    svc = RetrievalService()
    svc.ingest_paper(person1_paper_dict)
    evidence = svc.query_evidence("Which model has best BLEU?", top_k=5)
    comparison = svc.query_comparison("Best accuracy?", paper_ids=["p1", "p2"], k_per_paper=3)

Usage (service layer):
    from src.service import RetrievalService
    svc = RetrievalService()
    svc.start()  # if using background threads, etc.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_pipeline import HybridPipeline
from src.query_router import classify_intent
from src.cross_ref_expander import expand_cross_references, ExpanderConfig
from src.data_loader import load_local_subset
from src.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

logger = logging.getLogger(__name__)


class RetrievalService:
    """High-level service wrapping the full retrieval pipeline."""

    def __init__(
        self,
        *,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        ce_model: str = "BAAI/bge-reranker-base",
        auto_load_corpus: bool = True,
        corpus_limit: Optional[int] = None,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.pipeline = HybridPipeline(
            dense=self.dense,
            sparse=self.sparse,
            ce_model=ce_model,
        )
        self._lock = threading.RLock()
        self._ingested_paper_ids: set[str] = set()
        self._initialized = False

        if auto_load_corpus:
            self._load_initial_corpus(corpus_limit)

    def _load_initial_corpus(self, limit: Optional[int] = None) -> None:
        """Load local papers without deleting an existing persistent index."""
        papers, _ = load_local_subset()
        if limit and len(papers) > limit:
            papers = papers[:limit]
        # Add synthetic for method keywords
        papers.append(self._synth_method_paper())
        # Chroma is persistent, while BM25 is rebuilt in memory. On a service
        # restart, upsert the known corpus instead of deleting unrelated
        # persistent records; this also makes a target-paper refresh safe.
        reset = self.dense.count() == 0
        self.pipeline.index_papers(papers, reset=reset)
        for p in papers:
            self._ingested_paper_ids.add(p["paper_id"])
        self._initialized = True
        logger.info(f"RetrievalService initialized with {len(papers)} papers, {len(self._ingested_paper_ids)} unique IDs")

    @staticmethod
    def _synth_method_paper() -> dict:
        return {
            "paper_id": "hybrid_bench_001",
            "title": "Hybrid Lexical+Semantic Retrieval with RRF and Reranking",
            "elements": [
                {
                    "element_id": "h_t0",
                    "type": "text",
                    "section": "3. Method",
                    "page": 3,
                    "content": "We use RRF to fuse ranks and bge-reranker-base as a cross-encoder to re-rank candidates for better precision. k_rrf is set to 60 and we rerank the fused top-20.",
                },
                {
                    "element_id": "h_t1",
                    "type": "text",
                    "section": "4. Results",
                    "page": 4,
                    "content": "Our hybrid model combines BM25 lexical signals with dense embeddings from BGE-small. We report BLEU, ROUGE-L, and accuracy on standard benchmarks.",
                },
                {
                    "element_id": "h_tbl0",
                    "type": "table",
                    "section": "4. Results",
                    "page": 4,
                    "content": "| Model | BLEU | ROUGE-L | Accuracy |\n|---|---|---|---|\n| BM25-only | 24.1 | 47.8 | 68.5 |\n| Dense-only | 27.6 | 51.2 | 74.3 |\n| Hybrid+Rerank | 31.2 | 55.0 | 79.8 |",
                    "caption": "Table A: Hybrid achieves best BLEU and accuracy.",
                    "mentions": ["Table A shows Hybrid+Rerank yields best BLEU and accuracy."],
                },
                {
                    "element_id": "h_fig0",
                    "type": "figure",
                    "section": "3.2 Architecture",
                    "page": 3,
                    "caption": "Figure B: Hybrid pipeline overview (BM25 + Dense + RRF + bge-reranker).",
                    "mentions": ["Figure B illustrates the fusion and reranking stages."],
                },
            ],
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_paper(self, paper_dict: dict) -> int:
        """Ingest and index a single parsed paper from Person 1.

        Args:
            paper_dict: Paper in Person 1 schema (paper_id, title, elements).

        Returns:
            Number of chunks created and indexed.
        """
        if not isinstance(paper_dict, dict) or "paper_id" not in paper_dict:
            raise ValueError("paper_dict must be a dict with 'paper_id'")
        if "elements" not in paper_dict:
            raise ValueError("paper_dict must have 'elements' list")

        with self._lock:
            # Ensure pipeline is initialized
            if not self._initialized:
                self._load_initial_corpus()

            nd, ns = self.pipeline.index_papers([paper_dict], reset=False)
            self._ingested_paper_ids.add(paper_dict["paper_id"])
            logger.info(f"Ingested paper {paper_dict['paper_id']}: dense={nd}, sparse={ns}")
            return nd + ns  # total chunks (approximately)

    def query_evidence(
        self,
        query: str,
        paper_ids: Optional[Sequence[str]] = None,
        top_k: int = 5,
        *,
        use_router: bool = True,
        use_expander: bool = True,
    ) -> list[dict]:
        """Full hybrid retrieval returning EvidenceItems per schema.md.

        Args:
            query: Natural language question.
            paper_ids: Optional list of paper_ids to restrict search.
            top_k: Number of final reranked results to return.
            use_router: Enable intent classification + score boosting.
            use_expander: Enable cross-reference expansion.

        Returns:
            List of EvidenceItem dicts (rank 1..top_k).
        """
        if not self._initialized:
            self._load_initial_corpus()

        hits, _ = self.pipeline.search(
            query,
            k=top_k,
            paper_ids=paper_ids,
            use_router=use_router,
            use_expander=use_expander,
        )
        # Ensure strict schema.md EvidenceItem shape
        evidence = []
        for h in hits:
            ev = {
                "rank": h.get("rank"),
                "score": h.get("score") or h.get("ce_score"),
                "paper_id": h.get("paper_id"),
                "element_id": h.get("element_id"),
                "type": h.get("type"),
                "page": h.get("page"),
                "section": h.get("section", ""),
                "content": h.get("content", ""),
                "caption": h.get("caption"),
                "context": h.get("context", ""),
                "image_path": h.get("image_path") if h.get("image_path") not in (None, "") else None,
                "table_data": h.get("table_data"),
                "metadata": h.get("metadata") or {},
            }
            evidence.append(ev)
        return evidence

    def query_comparison(
        self,
        query: str,
        paper_ids: Sequence[str],
        k_per_paper: int = 3,
        *,
        use_router: bool = True,
        use_expander: bool = True,
    ) -> dict[str, list[dict]]:
        """Per-paper retrieval for Person 4's cross-paper comparison matrix.

        Runs independent search per paper and returns a dictionary mapping
        paper_id -> list of EvidenceItems (ranked 1..k_per_paper).

        Args:
            query: Natural language question.
            paper_ids: List of paper_ids to query (required).
            k_per_paper: Max results per paper.
            use_router: Enable intent classification + boosting per paper.
            use_expander: Enable cross-reference expansion per paper.

        Returns:
            dict[paper_id] -> list[EvidenceItem].
        """
        if not self._initialized:
            self._load_initial_corpus()

        if not paper_ids:
            raise ValueError("paper_ids must be provided for comparison query")

        per_paper = self.pipeline.retrieve_for_papers(
            query,
            paper_ids=paper_ids,
            k_per_paper=k_per_paper,
            use_router=use_router,
            use_expander=use_expander,
        )
        # Normalize to strict EvidenceItem shape
        out: dict[str, list[dict]] = {}
        for pid, hits in per_paper.items():
            out[pid] = [
                {
                    "rank": h.get("rank"),
                    "score": h.get("score") or h.get("ce_score"),
                    "paper_id": h.get("paper_id"),
                    "element_id": h.get("element_id"),
                    "type": h.get("type"),
                    "page": h.get("page"),
                    "section": h.get("section", ""),
                    "content": h.get("content", ""),
                    "caption": h.get("caption"),
                    "context": h.get("context", ""),
                    "image_path": h.get("image_path") if h.get("image_path") not in (None, "") else None,
                    "table_data": h.get("table_data"),
                    "metadata": h.get("metadata") or {},
                }
                for h in hits
            ]
        return out

    def health_check(self) -> dict[str, Any]:
        """Health check for service monitoring."""
        dense_count = self.dense.collection.count() if self.dense and self.dense.collection else 0
        sparse_count = self.sparse.count if self.sparse else 0
        return {
            "status": "healthy" if dense_count > 0 and sparse_count > 0 else "degraded",
            "dense_chunks": dense_count,
            "sparse_chunks": sparse_count,
            "ingested_papers": len(self._ingested_paper_ids),
            "initialized": self._initialized,
        }
