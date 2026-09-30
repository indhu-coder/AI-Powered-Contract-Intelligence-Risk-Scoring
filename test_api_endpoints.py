"""ContractIQ - FastAPI Endpoints Integration Tests.

Validates end-to-end functionality of contract uploads,
asynchronous analysis, metadata retrieval, and semantic search.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import CONTRACTS_DB, ANALYSES_DB, app

client = TestClient(app)
BASE_DIR = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def clean_db():
    CONTRACTS_DB.clear()
    ANALYSES_DB.clear()
    yield


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_documents" in data


def test_contract_upload_txt():
    content = b"This Consulting Agreement is dated February 1, 2020 between Company and Contractor. Non-compete applies."
    response = client.post(
        "/api/contracts/upload?auto_analyze=false",
        files={"file": ("contract.txt", content, "text/plain")},
    )
    assert response.status_code == 201
    data = response.json()
    assert "contract_id" in data
    assert data["filename"] == "contract.txt"
    assert data["status"] == "PENDING"
    assert data["word_count"] > 0


def test_contract_upload_and_sync_analyze():
    content = (
        b"CONSULTING AGREEMENT. Effective Date: January 15, 2022.\n"
        b"1. Non-Compete: Contractor shall not compete with Company for 2 years.\n"
        b"2. Exclusivity: Contractor agrees not to provide services to any other third party.\n"
        b"3. Governing Law: This agreement shall be governed by the laws of New York."
    )
    # Upload
    upload_resp = client.post(
        "/api/contracts/upload?auto_analyze=false",
        files={"file": ("nda_contract.txt", content, "text/plain")},
    )
    assert upload_resp.status_code == 201
    contract_id = upload_resp.json()["contract_id"]

    # Trigger synchronous analysis
    analyze_resp = client.post(f"/api/contracts/{contract_id}/analyze?sync=true")
    assert analyze_resp.status_code == 200
    analysis = analyze_resp.json()

    assert analysis["contract_id"] == contract_id
    assert analysis["overall_risk_score"] > 0
    assert analysis["overall_risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert len(analysis["clauses"]) > 0

    # Retrieve analysis via GET
    get_analysis = client.get(f"/api/contracts/{contract_id}/analysis")
    assert get_analysis.status_code == 200
    assert get_analysis.json()["overall_risk_score"] == analysis["overall_risk_score"]

    # Retrieve metadata via GET
    get_meta = client.get(f"/api/contracts/{contract_id}")
    assert get_meta.status_code == 200
    assert get_meta.json()["status"] == "COMPLETED"


def test_list_contracts():
    # Upload 2 contracts
    client.post(
        "/api/contracts/upload?auto_analyze=false",
        files={"file": ("c1.txt", b"First contract text here.", "text/plain")},
    )
    client.post(
        "/api/contracts/upload?auto_analyze=false",
        files={"file": ("c2.txt", b"Second contract text here.", "text/plain")},
    )

    response = client.get("/api/contracts")
    assert response.status_code == 200
    contracts = response.json()
    assert len(contracts) == 2


def test_semantic_search_endpoint():
    response = client.post(
        "/api/search/semantic",
        json={"query": "termination for convenience", "top_k": 3},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    if results:
        assert "similarity_score" in results[0]
        assert "contract_title" in results[0]


def test_upload_sample_pdf_if_present():
    pdf_path = BASE_DIR / "sample_contract.pdf"
    if not pdf_path.exists():
        pytest.skip("sample_contract.pdf not present in root.")

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/api/contracts/upload?auto_analyze=false",
            files={"file": ("sample_contract.pdf", f.read(), "application/pdf")},
        )
    assert response.status_code == 201
    data = response.json()
    assert data["character_count"] > 1000
