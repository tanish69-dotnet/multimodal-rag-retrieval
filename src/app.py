"""FastAPI service wrapper for the RetrievalService.

Provides REST endpoints for:
- POST /ingest: Ingest papers from Person 1 (schema.md input contract)
- POST /retrieve: Query evidence for Person 3 (schema.md output contract)
- POST /compare: Per-paper comparison for Person 4
- GET /health: Health check

Uses CORS middleware for cross-origin access (e.g., Streamlit/React frontends).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.service import RetrievalService

# Initialize app
app = FastAPI(
    title="Multimodal RAG Retrieval Service",
    description="Academic research assistant retrieval pipeline (Person 2)",
    version="1.0.0",
)

# CORS for Person 4's frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton service
_svc: RetrievalService | None = None


def get_service() -> RetrievalService:
    global _svc
    if _svc is None:
        _svc = RetrievalService(auto_load_corpus=True)
    return _svc


# ----------------------------------------------------------------------
# Pydantic Models (schema.md aligned)
# ----------------------------------------------------------------------
class Person1Element(BaseModel):
    """Element from Person 1's parsed paper."""
    element_id: str
    type: str  # "text" | "table" | "figure"
    section: str = ""
    page: Optional[int] = None
    content: str = ""
    caption: str = ""
    mentions: list[str] = Field(default_factory=list)
    image_path: Optional[str] = None
    # auxiliary fields
    source_image: Optional[str] = None
    figure_type: Optional[str] = None
    table_data: Optional[dict[str, Any]] = None
    source_url: Optional[str] = None
    source_location: Optional[str] = None


class Person1Paper(BaseModel):
    """Full paper from Person 1."""
    paper_id: str
    title: str = ""
    elements: list[Person1Element]


class IngestRequest(BaseModel):
    """POST /ingest request (single or batch)."""
    papers: list[Person1Paper]


class IngestResponse(BaseModel):
    """POST /ingest response."""
    ingested: int
    chunks_created: int
    paper_ids: list[str]


class EvidenceItem(BaseModel):
    """Output contract: Person 2 -> Person 3 (schema.md EvidenceItem)."""
    rank: int
    score: Optional[float] = None
    paper_id: Optional[str] = None
    element_id: Optional[str] = None
    type: Optional[str] = None
    page: Optional[int] = None
    section: str = ""
    content: str
    caption: Optional[str] = None
    context: str = ""
    image_path: Optional[str] = None
    table_data: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    """POST /retrieve request."""
    query: str
    paper_ids: Optional[list[str]] = None
    top_k: int = 5
    use_router: bool = True
    use_expander: bool = True


class RetrieveResponse(BaseModel):
    """POST /retrieve response."""
    evidence: list[EvidenceItem]


class CompareRequest(BaseModel):
    """POST /compare request."""
    query: str
    paper_ids: list[str]
    k_per_paper: int = 3
    use_router: bool = True
    use_expander: bool = True


class CompareResponse(BaseModel):
    """POST /compare response: dict[paper_id] -> list[EvidenceItem]."""
    results_by_paper: dict[str, list[EvidenceItem]]


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str
    dense_chunks: int
    sparse_chunks: int
    ingested_papers: int
    initialized: bool


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest one or more papers from Person 1."""
    svc = get_service()
    total_chunks = 0
    ingested_ids = []
    for p in req.papers:
        try:
            paper_dict = p.model_dump()
            chunks = svc.ingest_paper(paper_dict)
            total_chunks += chunks
            ingested_ids.append(p.paper_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to ingest {p.paper_id}: {e}")
    return IngestResponse(ingested=len(ingested_ids), chunks_created=total_chunks, paper_ids=ingested_ids)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    """Query evidence for Person 3 (Multi-LLM Reasoning)."""
    svc = get_service()
    try:
        evidence = svc.query_evidence(
            req.query,
            paper_ids=req.paper_ids,
            top_k=req.top_k,
            use_router=req.use_router,
            use_expander=req.use_expander,
        )
        return RetrieveResponse(evidence=[EvidenceItem(**e) for e in evidence])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Per-paper comparison for Person 4."""
    svc = get_service()
    try:
        by_paper = svc.query_comparison(
            req.query,
            paper_ids=req.paper_ids,
            k_per_paper=req.k_per_paper,
            use_router=req.use_router,
            use_expander=req.use_expander,
        )
        # Validate response model
        validated = {}
        for pid, hits in by_paper.items():
            validated[pid] = [EvidenceItem(**h) for h in hits]
        return CompareResponse(results_by_paper=validated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    svc = get_service()
    hc = svc.health_check()
    return HealthResponse(**hc)


class IngestPdfResponse(BaseModel):
    """POST /ingest-pdf response (Phase 9L)."""

    paper_id: str
    title: str = ""
    elements: int = 0
    text_elements: int = 0
    table_elements: int = 0
    figure_elements: int = 0
    figures_with_bytes: int = 0
    chunks_created: int = 0
    warnings: list[str] = Field(default_factory=list)


@app.post("/ingest-pdf", response_model=IngestPdfResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    paper_id: str | None = Form(default=None),
    title: str | None = Form(default=None),
):
    """Ingest a PDF file directly (Phase 9L, Person 1 scope).

    Validates ``%PDF-`` magic + size, parses text/tables/figures with PyMuPDF
    (no GPU, no OCR), then indexes through the existing ``ingest_paper``
    path (paper-scoped replace, other papers untouched).
    """
    from src.ingest.pdf_ingest import MAX_PDF_BYTES, PdfIngestError, parse_pdf_bytes

    raw = await file.read()
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds {MAX_PDF_BYTES} bytes",
        )
    if not raw[:5] == b"%PDF-":
        raise HTTPException(status_code=400, detail="not a PDF file (missing %PDF- magic)")

    svc = get_service()
    try:
        paper, warnings = parse_pdf_bytes(raw, paper_id=paper_id, title=title)
    except PdfIngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        chunks = svc.ingest_paper(paper)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    elements = paper.get("elements", [])
    figures_with_bytes = sum(
        1 for e in elements if e.get("type") == "figure" and e.get("image_path")
    )
    return IngestPdfResponse(
        paper_id=paper["paper_id"],
        title=paper.get("title", ""),
        elements=len(elements),
        text_elements=sum(1 for e in elements if e.get("type") == "text"),
        table_elements=sum(1 for e in elements if e.get("type") == "table"),
        figure_elements=sum(1 for e in elements if e.get("type") == "figure"),
        figures_with_bytes=figures_with_bytes,
        chunks_created=chunks,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=False)
