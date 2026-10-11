"""BM25 sparse retriever over the same multimodal corpus (text+table+figure).

Indexing uses the same chunk set as ``src.dense_retriever`` via
``chunk_paper`` + multimodal processors (via chunker). ``retrieve`` returns
matches with BM25 scores and their full document metadata (suitable for RRF
fusion and EvidenceItem reconstruction).

Usage:
    from src.data_loader import load_local_subset
    from src.sparse_retriever import SparseRetriever
    papers, _ = load_local_subset()
    papers.append(synth)
    sr = SparseRetriever()
    sr.index_papers(papers)
    sr.retrieve("BLEU score accuracy best model", k=15)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from rank_bm25 import BM25Okapi

from src.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_paper

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, strip punctuation, split on non-word chars."""
    if not text:
        return []
    text = text.lower().strip()
    return [t for t in re.split(r"\W+", text) if t]


class SparseRetriever:
    """BM25 (BM25Okapi) retriever built from the same chunks."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.corpus = []  # list of chunk dicts (same as dense)
        self.bm25: Optional[BM25Okapi] = None

    @property
    def count(self) -> int:
        return len(self.corpus)

    def index_papers(
        self,
        papers_data: Sequence[dict],
        *,
        reset: bool = True,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> int:
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        self.chunk_size = chunk_size
        self.overlap = overlap

        chunks: list[dict] = []
        for paper in papers_data:
            chunks.extend(chunk_paper(paper, chunk_size=chunk_size, overlap=overlap))

        if reset:
            self.corpus = chunks
        else:
            incoming_paper_ids = {chunk.get("paper_id") for chunk in chunks}
            self.corpus = [
                chunk for chunk in self.corpus
                if chunk.get("paper_id") not in incoming_paper_ids
            ]
            self.corpus.extend(chunks)
        if self.corpus:
            self.bm25 = BM25Okapi([tokenize(ch["content"]) for ch in self.corpus])
        else:
            self.bm25 = None
        print(f"[sparse.index] {len(papers_data)} papers -> {len(chunks)} chunks indexed (BM25Okapi)")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        paper_ids: Optional[Sequence[str]] = None,
        k: int = 15,
    ) -> list[dict]:
        query = (query or "").strip()
        if not query or not self.bm25 or not self.corpus:
            return []
        q = tokenize(query)
        if not q:
            return []

        # Full corpus scores; if paper_ids filter, re-score is cheap but easier:
        # rank by BM25 across all, then filter after. Alternatively compute
        # document-mask. For small corpus this is trivial.
        # rank_bm25.BM25Okapi has no 'score' kwarg and we need per-doc scores.
        try:
            scores_arr = self.bm25.get_scores(q)
        except Exception:
            scores_arr = None

        pairs = []
        for idx, ch in enumerate(self.corpus):
            if scores_arr is not None and hasattr(scores_arr, "__getitem__"):
                try:
                    s = float(scores_arr[idx])
                except Exception:
                    s = float(self.bm25.get_score(q, idx))
            else:
                s = float(self.bm25.get_score(q, idx))
            pairs.append((s, idx, ch))
        pairs.sort(key=lambda x: x[0], reverse=True)

        hits = []
        pidset = set(paper_ids) if paper_ids else None
        for s, idx, ch in pairs:
            if s <= 0 and self.count > 0 and len(hits) < 1:  # still record best
                pass  # keep
            if pidset and (ch.get("paper_id") not in pidset):
                continue
            hits.append({
                "rank_bm25": len(hits) + 1,
                "bm25_score": round(s, 6),
                "paper_id": ch.get("paper_id"),
                "element_id": ch.get("element_id"),
                "type": ch.get("type"),
                "page": ch.get("page"),
                "section": ch.get("section", ""),
                "chunk_id": ch.get("chunk_id"),
                "title": ch.get("title", ""),
                "caption": ch.get("caption"),
                "content": ch["content"],
                "context": ch.get("context", "") or "",
                "image_path": ch.get("image_path"),
                "source_image": ch.get("source_image"),
                "figure_type": ch.get("figure_type"),
                "raw_text": ch.get("raw_text", ""),
                "table_data": ch.get("table_data"),
                "metadata": ch.get("metadata") or {},
                "chunk": ch,  # keep for fusion
            })
            if len(hits) >= k:
                break
        return hits
