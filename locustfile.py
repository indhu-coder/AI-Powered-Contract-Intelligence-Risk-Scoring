"""ContractIQ - Locust Enterprise Load Testing Scenario.

Simulates enterprise legal and compliance officers interacting with the API:
- Health checks
- Document uploads
- Asynchronous contract analysis polling
- FAISS dense semantic searches

Run:
    locust -f locustfile.py --host http://127.0.0.1:8000
"""
from __future__ import annotations

import random
import time
from locust import HttpUser, between, task

SAMPLE_CLAUSES = [
    "termination for convenience with 30 days notice",
    "uncapped liability and indemnification obligations",
    "exclusive license grant and territory restrictions",
    "intellectual property assignment of inventions and patents",
    "governing law and dispute resolution in Delaware",
    "non-compete and non-solicitation restrictions on employees",
    "confidentiality duration and survival after termination",
    "audit rights and financial inspection records",
]

SAMPLE_CONTRACT_CONTENT = b"""
EXECUTIVE CONSULTING AGREEMENT
This Agreement is entered into on March 1, 2024.
1. SERVICES: Consultant shall provide software architecture advisory services.
2. INDEMNITY: Client agrees to hold Consultant harmless against all claims, without limitation.
3. GOVERNING LAW: The laws of the State of New York govern this Agreement.
4. TERMINATION: Either party may terminate with 15 days written notice.
5. CONFIDENTIALITY: Proprietary information shall remain confidential for 3 years.
"""


class LegalComplianceUser(HttpUser):
    """Simulates a legal reviewer using the ContractIQ platform."""
    wait_time = between(1, 3)

    @task(5)
    def check_health(self):
        """Monitors system health and index metrics."""
        self.client.get("/health", name="/health")

    @task(3)
    def search_vector_store(self):
        """Performs semantic clause search."""
        query = random.choice(SAMPLE_CLAUSES)
        payload = {
            "query": query,
            "top_k": 5,
            "min_score": 0.15,
        }
        self.client.post("/api/search/semantic", json=payload, name="/api/search/semantic")

    @task(2)
    def list_contracts(self):
        """Lists uploaded contracts."""
        self.client.get("/api/contracts", name="/api/contracts")

    @task(1)
    def upload_and_analyze_contract(self):
        """Uploads a contract and triggers risk analysis."""
        files = {
            "file": (
                f"locust_contract_{int(time.time())}.txt",
                SAMPLE_CONTRACT_CONTENT,
                "text/plain",
            )
        }
        with self.client.post(
            "/api/contracts/upload?auto_analyze=true",
            files=files,
            name="/api/contracts/upload",
            catch_response=True,
        ) as response:
            if response.status_code == 201:
                contract_id = response.json().get("contract_id")
                # Poll status
                if contract_id:
                    self.client.get(f"/api/contracts/{contract_id}", name="/api/contracts/{id}")
            else:
                response.failure(f"Upload failed with status {response.status_code}")
