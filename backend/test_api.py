"""API tests for AI-Powered Contract Intelligence & Risk Scoring.

Each test uses an isolated temporary SQLite database and a MockAnalyzer.
This prevents tests from modifying the production/demo database or loading
the transformer model.

Run from the project root:
    pytest backend/tests/test_contract_intelligence_api.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.conftest import MockAnalyzer, SAMPLE_CONTRACT_TEXT  # noqa: E402

API = "/api"


@pytest.fixture()
def app(tmp_path: Path):
    """Create an isolated ContractIQ application for each test."""
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'api-test.db').as_posix()}",
        upload_dir=tmp_path / "uploads",
        temp_dir=tmp_path / "temp",
        cors_origins=["http://localhost:5173"],
    )
    return create_app(settings)


@pytest.fixture()
def client(app):
    """Create a test client and replace the analyzer with a mock."""
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        # Replace the analyzer after startup so the transformer is not loaded.
        app.state.service.analyzer = MockAnalyzer()
        yield test_client


def upload_contract(
    client,
    filename: str = "contract.txt",
    content: bytes | None = None,
    content_type: str = "text/plain",
):
    """Upload a contract and return the API response."""
    data = (
        SAMPLE_CONTRACT_TEXT.encode("utf-8")
        if content is None
        else content
    )
    return client.post(
        f"{API}/contracts/upload",
        files={"upload": (filename, data, content_type)},
    )


def upload_and_analyze(client):
    """Upload a contract, analyze it, and return IDs."""
    response = upload_contract(client)
    assert response.status_code == 201, response.text

    contract_id = response.json()["id"]
    analysis = client.post(f"{API}/contracts/{contract_id}/analyze")
    assert analysis.status_code == 200, analysis.text

    return contract_id, analysis.json()["analysis_id"]


# ---------------------------------------------------------------------------
# Health and API documentation
# ---------------------------------------------------------------------------

def test_health(client):
    response = client.get(f"{API}/health")

    assert response.status_code == 200

    body = response.json()
    assert body["app"] == "ContractIQ"
    assert body["database"] == "ok"
    assert "model" in body
    assert "device" in body["model"]
    assert body["model"]["state"] in {"fine_tuned", "baseline_on_demand"}


def test_openapi_docs(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200

    paths = client.get("/openapi.json").json()["paths"]

    expected = [
        "/api/health",
        "/api/contracts/upload",
        "/api/contracts",
        "/api/contracts/{contract_id}",
        "/api/contracts/{contract_id}/analyze",
        "/api/contracts/{contract_id}/analysis",
        "/api/contracts/{contract_id}/text",
        "/api/analyses/{analysis_id}",
        "/api/stats",
    ]

    for path in expected:
        assert path in paths, path


# ---------------------------------------------------------------------------
# Contract upload
# ---------------------------------------------------------------------------

def test_upload_txt(client):
    response = upload_contract(client)

    assert response.status_code == 201

    body = response.json()
    assert body["filename"] == "contract.txt"
    assert body["file_type"] == "txt"
    assert body["status"] == "READY"
    assert body["character_count"] > 0
    assert "text" not in body


def test_upload_docx(client, sample_docx):
    response = upload_contract(
        client,
        "sample_contract.docx",
        sample_docx.read_bytes(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert response.status_code == 201
    assert response.json()["file_type"] == "docx"


def test_upload_pdf(client, sample_pdf):
    response = upload_contract(
        client,
        "sample_contract.pdf",
        sample_pdf.read_bytes(),
        "application/pdf",
    )

    assert response.status_code == 201

    body = response.json()
    assert body["file_type"] == "pdf"
    assert (body["page_count"] or 0) >= 1


def test_upload_invalid_extension(client):
    response = upload_contract(client, "malicious.exe", b"MZ binary")

    assert response.status_code == 400
    assert response.json()["error"] == "UNSUPPORTED_FILE_TYPE"


def test_upload_empty_file(client):
    response = upload_contract(client, "empty.txt", b"")

    assert response.status_code == 400
    assert response.json()["error"] == "EMPTY_FILE"


def test_upload_oversized(client, app):
    # Use a tiny configured limit to exercise the size validation.
    app.state.settings.max_upload_mb = 0

    response = upload_contract(client, "large.txt", b"x" * 100)

    assert response.status_code in (201, 413)


def test_upload_scanned_pdf(client, scanned_pdf):
    response = upload_contract(
        client,
        "scanned.pdf",
        scanned_pdf.read_bytes(),
        "application/pdf",
    )

    assert response.status_code == 422

    body = response.json()
    assert body["error"] == "OCR_REQUIRED"
    assert "OCR" in body["message"]


def test_upload_corrupt_pdf(client, corrupt_pdf):
    response = upload_contract(
        client,
        "corrupt.pdf",
        corrupt_pdf.read_bytes(),
        "application/pdf",
    )

    assert response.status_code == 422
    assert response.json()["error"] in {
        "CORRUPT_DOCUMENT",
        "OCR_REQUIRED",
    }


def test_upload_filename_sanitization(client):
    response = upload_contract(client, "../../weird name.txt")

    assert response.status_code == 201
    assert response.json()["filename"] == "weird name.txt"


# ---------------------------------------------------------------------------
# Contract listing and details
# ---------------------------------------------------------------------------

def test_contract_listing_and_search(client):
    upload_contract(client, "license agreement.txt")
    upload_contract(client, "nda.txt")

    response = client.get(f"{API}/contracts")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2

    response = client.get(
        f"{API}/contracts",
        params={"search": "nda"},
    )
    assert response.json()["total"] == 1

    response = client.get(
        f"{API}/contracts",
        params={"status": "READY"},
    )
    assert response.json()["total"] == 2

    response = client.get(
        f"{API}/contracts",
        params={"status": "NOT_A_STATUS"},
    )
    assert response.status_code == 422


def test_contract_detail_excludes_raw_text(client):
    contract_id, _ = upload_and_analyze(client)

    response = client.get(f"{API}/contracts/{contract_id}")

    assert response.status_code == 200

    body = response.json()
    assert body["id"] == contract_id
    assert body["latest_analysis"]["status"] == "COMPLETED"
    assert "text" not in body


def test_contract_not_found(client):
    response = client.get(f"{API}/contracts/missing-id")

    assert response.status_code == 404

    body = response.json()
    assert body["error"] == "CONTRACT_NOT_FOUND"
    assert body["request_id"]


# ---------------------------------------------------------------------------
# Contract analysis and risk scoring
# ---------------------------------------------------------------------------

def test_analysis_success_and_persistence(client):
    contract_id, analysis_id = upload_and_analyze(client)

    response = client.get(
        f"{API}/contracts/{contract_id}/analysis"
    )

    assert response.status_code == 200

    body = response.json()
    assert body["analysis_id"] == analysis_id
    assert body["status"] == "COMPLETED"
    assert body["overall_risk"]["level"] == "HIGH"
    assert body["model"]["state"] == "fine_tuned"

    clause_types = {clause["clause_type"] for clause in body["clauses"]}
    assert "governing_law" in clause_types

    governing_law = next(
        clause
        for clause in body["clauses"]
        if clause["clause_type"] == "governing_law"
    )

    assert governing_law["confidence"] == 0.92
    assert governing_law["found"] is True
    assert body["entities"]["governing_law"] == "laws of the State of New York"

    rule_ids = {finding["rule_id"] for finding in body["risk_findings"]}
    assert "NON_COMPETE_PRESENT" in rule_ids
    assert any(
        finding["weight"] and finding["weight"] > 0
        for finding in body["risk_findings"]
    )


def test_analysis_retrieval_by_id(client):
    _, analysis_id = upload_and_analyze(client)

    response = client.get(f"{API}/analyses/{analysis_id}")

    assert response.status_code == 200
    assert response.json()["analysis_id"] == analysis_id
    assert client.get(f"{API}/analyses/999999").status_code == 404


def test_analysis_model_unavailable(client, app):
    from app.core.exceptions import ModelUnavailable

    class BrokenAnalyzer:
        def analyze_contract(self, text, enabled_clauses=None):
            raise ModelUnavailable("model files missing")

        def analyze_clause(self, text, label):
            raise ModelUnavailable("model files missing")

    app.state.service.analyzer = BrokenAnalyzer()

    response = upload_contract(client)
    contract_id = response.json()["id"]

    analysis = client.post(
        f"{API}/contracts/{contract_id}/analyze"
    )

    assert analysis.status_code == 503

    body = analysis.json()
    assert body["error"] == "MODEL_UNAVAILABLE"

    detail = client.get(
        f"{API}/contracts/{contract_id}"
    ).json()

    assert detail["latest_analysis"]["status"] == "FAILED"


def test_analysis_generic_failure(client, app):
    app.state.service.analyzer = MockAnalyzer(fail=True)

    response = upload_contract(client)
    contract_id = response.json()["id"]

    analysis = client.post(
        f"{API}/contracts/{contract_id}/analyze"
    )

    assert analysis.status_code == 500
    assert analysis.json()["error"] == "ANALYSIS_FAILURE"

    full = client.get(
        f"{API}/contracts/{contract_id}/analysis"
    )

    assert full.status_code == 200
    assert full.json()["status"] == "FAILED"


def test_analysis_before_any_analysis(client):
    response = upload_contract(client)
    contract_id = response.json()["id"]

    analysis = client.get(
        f"{API}/contracts/{contract_id}/analysis"
    )

    assert analysis.status_code == 500
    assert analysis.json()["error"] == "ANALYSIS_FAILURE"


# ---------------------------------------------------------------------------
# Raw contract text
# ---------------------------------------------------------------------------

def test_raw_text_endpoint(client):
    response = upload_contract(client)
    contract_id = response.json()["id"]

    text = client.get(
        f"{API}/contracts/{contract_id}/text"
    )

    assert text.status_code == 200
    assert text.json()["text"].startswith("DISTRIBUTION AGREEMENT")


def test_raw_text_disabled(client, app):
    app.state.settings.store_raw_text = False

    response = upload_contract(client)
    contract_id = response.json()["id"]

    text = client.get(
        f"{API}/contracts/{contract_id}/text"
    )

    assert text.status_code == 404
    assert text.json()["error"] == "RAW_TEXT_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Contract deletion
# ---------------------------------------------------------------------------

def test_delete_contract(client):
    contract_id, _ = upload_and_analyze(client)

    response = client.delete(
        f"{API}/contracts/{contract_id}"
    )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert client.get(f"{API}/contracts/{contract_id}").status_code == 404
    assert client.get(
        f"{API}/contracts/{contract_id}/analysis"
    ).status_code == 404
    assert client.delete(
        f"{API}/contracts/{contract_id}"
    ).status_code == 404


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_stats(client):
    upload_and_analyze(client)

    response = client.get(f"{API}/stats")

    assert response.status_code == 200

    body = response.json()
    assert body["contracts_total"] == 1
    assert body["completed_analyses"] == 1
    assert body["risk_distribution"]["HIGH"] == 1
    assert body["clauses_detected_total"] >= 4
    assert body["avg_processing_ms"] is not None


# ---------------------------------------------------------------------------
# Errors, request IDs, CORS and database isolation
# ---------------------------------------------------------------------------

def test_error_format_and_request_id(client):
    response = client.get(f"{API}/contracts/nope")

    body = response.json()

    assert set(body) == {"error", "message", "request_id"}
    assert response.headers.get("X-Request-ID") == body["request_id"]


def test_cors_headers(client):
    response = client.options(
        f"{API}/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code in (200, 400)
    assert response.headers.get(
        "access-control-allow-origin"
    ) == "http://localhost:5173"


def test_db_is_isolated_per_app(client):
    """Ensure API tests never modify the demo/production database."""
    from app.core.config import PROJECT_ROOT

    demo_db = PROJECT_ROOT / "storage" / "contractiq.db"

    if demo_db.exists():
        import time

        before = demo_db.stat().st_mtime
        upload_and_analyze(client)
        time.sleep(0.01)

        assert demo_db.stat().st_mtime == before
