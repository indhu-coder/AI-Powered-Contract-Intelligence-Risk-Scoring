"""ContractIQ - Production FastAPI REST API Backend.

Provides RESTful endpoints for contract ingestion, asynchronous NLP analysis,
risk scoring, and FAISS semantic vector search.

Run locally:
    uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import io
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import logging_config
from contract_risk_engine import ContractIntelligenceEngine
from tasks import run_contract_analysis
from vector_store import VectorStore

logger = logging_config.get_logger("api")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ContractIQ API",
    version="1.0.0",
    description=(
        "AI-Powered Contract Intelligence & Legal Risk Scoring Platform. "
        "Extracts clauses, identifies high-risk commitments, and performs FAISS vector search."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_and_timing_middleware(request: Request, call_next):
    """Correlates requests with an X-Request-ID and tracks latency in milliseconds."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.perf_counter()
    
    # Store request id in request state
    request.state.request_id = req_id

    response = await call_next(request)
    
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
    
    # Exclude root/docs/health from heavy logging
    if not request.url.path.startswith(("/docs", "/openapi.json", "/redoc")):
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)",
            extra={"request_id": req_id},
        )
    return response

# In-memory storage for contracts and analyses (persisted per process)
CONTRACTS_DB: Dict[str, Dict[str, Any]] = {}
ANALYSES_DB: Dict[str, Dict[str, Any]] = {}

# Lazy-loaded vector store and risk engine singletons
_vector_store: Optional[VectorStore] = None
_risk_engine: Optional[ContractIntelligenceEngine] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        if VECTOR_DIR.exists():
            _vector_store.load(VECTOR_DIR)
    return _vector_store


def get_risk_engine() -> ContractIntelligenceEngine:
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = ContractIntelligenceEngine()
    return _risk_engine


# --- Pydantic Request & Response Schemas ---

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Legal query or requirement to search for")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum cosine similarity threshold")


class SemanticSearchResult(BaseModel):
    id: str
    contract_title: str
    text: str
    similarity_score: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ContractUploadResponse(BaseModel):
    contract_id: str
    filename: str
    file_type: str
    character_count: int
    word_count: int
    status: str
    message: str


class ContractMetadataResponse(BaseModel):
    contract_id: str
    filename: str
    file_type: str
    character_count: int
    word_count: int
    status: str
    uploaded_at: str
    overall_risk_score: Optional[int] = None
    overall_risk_level: Optional[str] = None


class RiskFindingItem(BaseModel):
    rule_id: str
    clause_type: str
    title: str
    description: str
    weight: int
    severity: str
    recommendation: str


class ClauseMatchItem(BaseModel):
    clause_type: str
    display_name: str
    found: bool
    confidence: float
    excerpt: str
    risk_level: str


class ContractAnalysisResponse(BaseModel):
    contract_id: str
    contract_title: str
    character_count: int
    word_count: int
    overall_risk_score: int
    overall_risk_level: str
    extracted_dates: Dict[str, Optional[str]]
    durations: Dict[str, Optional[str]]
    entities: Dict[str, List[str]]
    clauses: List[ClauseMatchItem]
    risk_findings: List[RiskFindingItem]
    disclaimer: str


# --- Helper: Document Text Extraction ---

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = [page.get_text("text").strip() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        if len(text.strip()) < 20:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Scanned/image-only PDF detected with no extractable text layer.",
            )
        return text

    elif ext == ".docx":
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    elif ext in (".txt", ".text"):
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore")

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported file type '{ext}'. Supported formats: .pdf, .docx, .txt",
    )


def execute_background_analysis(contract_id: str):
    """Worker task executed asynchronously."""
    contract = CONTRACTS_DB.get(contract_id)
    if not contract:
        logger.warning(f"Background analysis requested for non-existent contract {contract_id}")
        return

    logger.info(f"Starting background contract analysis for contract {contract_id} ({contract['filename']})")
    start_time = time.perf_counter()
    contract["status"] = "ANALYZING"
    try:
        analysis_result = run_contract_analysis(
            contract_id=contract_id,
            text=contract["text"],
            title=contract["filename"],
        )
        ANALYSES_DB[contract_id] = analysis_result
        contract["status"] = "COMPLETED"
        contract["overall_risk_score"] = analysis_result["overall_risk_score"]
        contract["overall_risk_level"] = analysis_result["overall_risk_level"]
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Completed analysis for {contract_id} in {elapsed:.2f}s "
            f"[Risk Score: {contract['overall_risk_score']}, Level: {contract['overall_risk_level']}]"
        )
    except Exception as exc:
        contract["status"] = "FAILED"
        contract["error"] = str(exc)
        logger.error(f"Analysis failed for contract {contract_id}: {exc}", exc_info=True)


# --- API Routes ---

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
def health_check():
    store = get_vector_store()
    return {
        "status": "healthy",
        "service": "ContractIQ API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_contracts": len(CONTRACTS_DB),
        "total_analyses": len(ANALYSES_DB),
        "vector_store_documents": store.count(),
    }


@app.get("/health/extended", tags=["System"])
def extended_health_check():
    """Returns detailed infrastructure and operational health metrics."""
    store = get_vector_store()
    model_path = BASE_DIR / "models" / "fine_tuned_clause_model"
    vector_path = VECTOR_DIR / "contracts.index"
    
    return {
        "status": "healthy",
        "service": "ContractIQ API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "storage": {
            "total_contracts_loaded": len(CONTRACTS_DB),
            "total_analyses_completed": sum(1 for c in CONTRACTS_DB.values() if c.get("status") == "COMPLETED"),
            "vector_store_indexed_passages": store.count(),
            "vector_index_exists": vector_path.exists(),
            "model_weights_exist": (model_path / "model.safetensors").exists(),
        },
        "features": {
            "faiss_dense_search": True,
            "spacy_ner_enabled": True,
            "extractive_qa_transformer": True,
            "risk_scoring_rules": 10,
        },
    }


@app.post(
    "/api/contracts/upload",
    response_model=ContractUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Contracts"],
)
async def upload_contract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_analyze: bool = Query(True, description="Automatically trigger asynchronous analysis after upload"),
):
    """Upload a contract document (PDF, DOCX, TXT) and register it for intelligence analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = extract_text(content, file.filename)
    contract_id = f"ctr_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    words = len(text.split())
    ext = Path(file.filename).suffix.lower().lstrip(".")

    # Persist file
    save_path = UPLOAD_DIR / f"{contract_id}_{file.filename}"
    save_path.write_bytes(content)

    CONTRACTS_DB[contract_id] = {
        "contract_id": contract_id,
        "filename": file.filename,
        "file_type": ext,
        "file_path": str(save_path),
        "character_count": len(text),
        "word_count": words,
        "text": text,
        "status": "PENDING",
        "uploaded_at": now_iso,
        "overall_risk_score": None,
        "overall_risk_level": None,
    }

    if auto_analyze:
        background_tasks.add_task(execute_background_analysis, contract_id)

    return ContractUploadResponse(
        contract_id=contract_id,
        filename=file.filename,
        file_type=ext,
        character_count=len(text),
        word_count=words,
        status="ANALYZING" if auto_analyze else "PENDING",
        message="Contract uploaded successfully. Analysis scheduled in background.",
    )


@app.post(
    "/api/contracts/{contract_id}/analyze",
    response_model=Dict[str, Any],
    tags=["Contracts"],
)
async def trigger_analysis(
    contract_id: str,
    background_tasks: BackgroundTasks,
    sync: bool = Query(False, description="Run synchronously and wait for analysis to complete"),
):
    """Trigger legal intelligence & risk scoring for an uploaded contract."""
    contract = CONTRACTS_DB.get(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found.")

    if sync:
        execute_background_analysis(contract_id)
        if contract["status"] == "FAILED":
            raise HTTPException(status_code=500, detail=f"Analysis failed: {contract.get('error')}")
        return ANALYSES_DB[contract_id]

    contract["status"] = "ANALYZING"
    background_tasks.add_task(execute_background_analysis, contract_id)
    return {
        "contract_id": contract_id,
        "status": "ANALYZING",
        "message": "Analysis initiated in background.",
    }


@app.get(
    "/api/contracts/{contract_id}",
    response_model=ContractMetadataResponse,
    tags=["Contracts"],
)
def get_contract_metadata(contract_id: str):
    """Retrieve metadata and current analysis status for a contract."""
    contract = CONTRACTS_DB.get(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found.")

    return ContractMetadataResponse(
        contract_id=contract["contract_id"],
        filename=contract["filename"],
        file_type=contract["file_type"],
        character_count=contract["character_count"],
        word_count=contract["word_count"],
        status=contract["status"],
        uploaded_at=contract["uploaded_at"],
        overall_risk_score=contract.get("overall_risk_score"),
        overall_risk_level=contract.get("overall_risk_level"),
    )


@app.get(
    "/api/contracts/{contract_id}/analysis",
    response_model=ContractAnalysisResponse,
    tags=["Contracts"],
)
def get_contract_analysis(contract_id: str):
    """Retrieve full risk analysis results, findings, and clause excerpts."""
    contract = CONTRACTS_DB.get(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found.")

    analysis = ANALYSES_DB.get(contract_id)
    if not analysis:
        if contract["status"] == "ANALYZING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Contract analysis is currently in progress. Please check again shortly.",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Analysis for contract '{contract_id}' has not been performed yet.",
        )

    return analysis


@app.get(
    "/api/contracts",
    response_model=List[ContractMetadataResponse],
    tags=["Contracts"],
)
def list_contracts(
    status_filter: Optional[str] = Query(None, description="Filter by status (PENDING, ANALYZING, COMPLETED, FAILED)"),
    risk_filter: Optional[str] = Query(None, description="Filter by risk level (LOW, MEDIUM, HIGH)"),
):
    """List all registered contracts."""
    contracts = list(CONTRACTS_DB.values())

    if status_filter:
        contracts = [c for c in contracts if c["status"].upper() == status_filter.upper()]

    if risk_filter:
        contracts = [c for c in contracts if (c.get("overall_risk_level") or "").upper() == risk_filter.upper()]

    return [
        ContractMetadataResponse(
            contract_id=c["contract_id"],
            filename=c["filename"],
            file_type=c["file_type"],
            character_count=c["character_count"],
            word_count=c["word_count"],
            status=c["status"],
            uploaded_at=c["uploaded_at"],
            overall_risk_score=c.get("overall_risk_score"),
            overall_risk_level=c.get("overall_risk_level"),
        )
        for c in contracts
    ]


@app.post(
    "/api/search/semantic",
    response_model=List[SemanticSearchResult],
    tags=["Semantic Search"],
)
def semantic_search(request: SemanticSearchRequest):
    """Perform FAISS dense vector search over the contract repository."""
    store = get_vector_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Run build_vector_index.py to populate vectors.",
        )

    results = store.search(
        query=request.query,
        top_k=request.top_k,
        min_score=request.min_score,
    )

    return [
        SemanticSearchResult(
            id=r["id"],
            contract_title=r.get("contract_title", "Unknown"),
            text=r["text"],
            similarity_score=r["similarity_score"],
            char_start=r.get("char_start"),
            char_end=r.get("char_end"),
        )
        for r in results
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
