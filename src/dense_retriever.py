"""Baseline dense retrieval over the schema.md corpus.

Persistent ChromaDB store under ``chroma_db/`` with
``BAAI/bge-small-en-v1.5`` embeddings (L2-normalized, cosine metric).

Pipeline
--------
  retriever = DenseRetriever()
  n_chunks  = retriever.index_papers(papers)     # chunk + embed + upsert
  hits      = retriever.retrieve(query, paper_ids=["..."], k=5)

Returned hits are EvidenceItem records per schema.md:

  {"rank", "score", "paper_id", "element_id", "type", "page", "section",
   "content", "context", ...}

``score`` is cosine similarity in [0, 1] (Chroma returns cosine distance
``1 - similarity``; we invert it).
"""

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from src.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_paper  # delegates visuals to src.multimodal_processor

MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "academic_rag_dense"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = ROOT / "chroma_db"


class BGEEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function wrapping sentence-transformers BGE."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(
            list(input), normalize_embeddings=True, show_progress_bar=False
        ).tolist()


def _sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool (None is rejected)."""
    out = {}
    for key, value in meta.items():
        if value is None:
            out[key] = ""
        elif isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float, str)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _int_or_none(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


class DenseRetriever:
    """ChromaDB-backed dense retriever with BGE embeddings."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        collection_name: str = COLLECTION_NAME,
        chroma_dir: Union[str, Path] = DEFAULT_CHROMA_DIR,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self._model = SentenceTransformer(model_name)
        self._embedding_fn = BGEEmbeddingFunction(self._model)
        self.collection = self._create_collection()

    # ------------------------------------------------------------------ #
    # Collection management
    # ------------------------------------------------------------------ #
    def _create_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        """Drop and recreate the collection so re-indexing is idempotent."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection does not exist yet
        self.collection = self._create_collection()

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_papers(
        self,
        papers_data: Sequence[dict],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        reset: bool = True,
    ) -> int:
        """Chunk, embed, and upsert every schema.md paper into ChromaDB.

        Args:
            papers_data: list of schema.md paper records
                         (e.g. from ``data/processed_papers.json``)
            chunk_size: target words per text chunk
            overlap: words shared between consecutive text chunks
            reset: drop existing collection before indexing (keeps the index
                   exactly in sync with the input corpus)

        Returns:
            number of chunks indexed.
        """
        if reset:
            self.reset_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        skipped = 0

        for paper in papers_data:
            chunks = chunk_paper(paper, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                skipped += 1
                continue
            for chunk in chunks:
                ids.append(chunk["chunk_id"])
                documents.append(chunk["content"])
                metadatas.append(_sanitize_metadata(self._chunk_metadata(chunk)))

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        print(f"[index] {len(papers_data)} papers -> {len(ids)} chunks "
              f"upserted to chroma:{self.collection_name} "
              f"(embedding dim affected by {self.model_name})")
        print(f"[index] papers skipped (no indexable text/captions): {skipped}")
        return len(ids)

    @staticmethod
    def _chunk_metadata(chunk: dict) -> dict:
        """Carry every field needed for schema.md EvidenceItem reconstruction."""
        meta = {
            "paper_id": chunk.get("paper_id", ""),
            "element_id": chunk.get("element_id", ""),
            "type": chunk.get("type", ""),
            "page": chunk.get("page"),
            "section": chunk.get("section", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "title": chunk.get("title", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "raw_text": chunk.get("raw_text", ""),
            # schema.md EvidenceItem
            "caption": chunk.get("caption", ""),
            "context": chunk.get("context", "") or "",
            "image_path": chunk.get("image_path"),
        }
        for key in ("source_image", "figure_type", "header", "mentions"):
            if chunk.get(key):
                meta[key] = chunk[key]
        if chunk.get("table_data"):
            meta["table_data_json"] = json.dumps(chunk["table_data"], ensure_ascii=False, sort_keys=True)
        for key in ("source_url", "source_location"):
            if chunk.get(key):
                meta[key] = chunk[key]
        # Preserve explicitly-empty caption/context so retrieval can surface them.
        if "caption" not in meta:
            meta["caption"] = chunk.get("caption", "")
        if "context" not in meta:
            meta["context"] = chunk.get("context", "") or ""
        if "image_path" not in meta:
            meta["image_path"] = chunk.get("image_path")
        return meta

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        paper_ids: Optional[Sequence[str]] = None,
        k: int = 5,
    ) -> list[dict]:
        """Embed the query and return the top-k EvidenceItems (schema.md).

        Args:
            query: natural-language question / search text
            paper_ids: optional whitelist of ``paper_id`` values to restrict
                       the search (metadata filter)
            k: number of hits to return (capped by collection size)

        Returns:
            ranked list of EvidenceItem dicts (descending similarity).
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        if self.collection.count() == 0:
            return []

        where = None
        if paper_ids:
            paper_ids = [p for p in paper_ids if p]
            if paper_ids:
                where = {"paper_id": {"$in": list(paper_ids)}}

        result = self.collection.query(
            query_texts=[query],
            n_results=max(1, k),
            where=where,
        )

        hits = []
        id_list = result.get("ids", [[]])[0]
        dist_list = result.get("distances", [[]])[0] or []
        doc_list = result.get("documents", [[]])[0] or []
        meta_list = result.get("metadatas", [[]])[0] or []

        for rank, chunk_id in enumerate(id_list, start=1):
            meta = meta_list[rank - 1] if rank - 1 < len(meta_list) else {}
            meta = meta or {}
            distance = dist_list[rank - 1] if rank - 1 < len(dist_list) else None
            document = doc_list[rank - 1] if rank - 1 < len(doc_list) else None

            score = None
            if distance is not None:
                score = round(1.0 - float(distance), 4)

            hits.append(self._to_evidence(rank, score, document, meta))

        return hits

    @staticmethod
    def _to_evidence(rank: int, score: Optional[float], document, meta: dict) -> dict:
        """Mold a Chroma hit into the strict schema.md EvidenceItem shape.

        Core keys match schema.md exactly (rank, score, paper_id, element_id,
        type, page, section, content, caption, context, image_path). Helpful
        extras (chunk_id, title, source_image, figure_type, mentions) are also
        attached.
        """
        # ``document`` is the full embed-ready string (preamble + markdown/caption)
        content_for_evidence = document or meta.get("content") or ""
        # Prefer the processor's ``context`` (joined mentions); fall back to
        # the chunk's raw text so figure captions remain visible.
        context_value = meta.get("context")
        if context_value is None:
            context_value = meta.get("raw_text") or ""
        table_data = None
        raw_table_data = meta.get("table_data_json")
        if isinstance(raw_table_data, str) and raw_table_data:
            try:
                parsed = json.loads(raw_table_data)
                table_data = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                table_data = None
        metadata = {
            key: meta[key]
            for key in ("source_url", "source_location")
            if meta.get(key)
        }
        return {
            # --- schema.md EvidenceItem (strict) ---
            "rank": rank,
            "score": score,
            "paper_id": meta.get("paper_id") or None,
            "element_id": meta.get("element_id") or None,
            "type": meta.get("type") or None,
            "page": _int_or_none(meta.get("page")),
            "section": meta.get("section") or "",
            "content": content_for_evidence,
            "caption": meta.get("caption") or None,
            "context": context_value or "",
            "image_path": meta.get("image_path") if meta.get("image_path") not in ("", None) else None,
            "table_data": table_data,
            "metadata": metadata,
            # --- helpful extras ---
            "chunk_id": meta.get("chunk_id") or None,
            "title": meta.get("title") or "",
            "source_image": meta.get("source_image") or None,
            "figure_type": meta.get("figure_type") or None,
            "mentions": meta.get("mentions") if isinstance(meta.get("mentions"), list) else [],
        }
