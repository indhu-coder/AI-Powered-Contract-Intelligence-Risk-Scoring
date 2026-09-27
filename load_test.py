"""ContractIQ - High-Concurrency Load Testing & Performance Benchmark Suite.

Simulates concurrent user traffic across:
1. Health & system readiness endpoints
2. Dense FAISS semantic vector search
3. End-to-end contract ingestion, OCR/text extraction, and risk analysis

Outputs detailed latency percentiles (p50, p90, p95, p99), QPS, and exports
benchmark metrics to reports/week4_load_test_report.json.

Run:
    python load_test.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Ensure console supports utf-8 safely on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORTS_DIR / "week4_load_test_report.json"

SEARCH_QUERIES = [
    "termination for convenience with 30 days notice",
    "uncapped liability and indemnification obligations",
    "exclusive license grant and territory restrictions",
    "intellectual property assignment of inventions and patents",
    "governing law and dispute resolution in Delaware",
    "non-compete and non-solicitation restrictions on employees",
    "confidentiality duration and survival after termination",
    "audit rights and financial inspection records",
]

SAMPLE_CONTRACT_TEXT = """
MASTER SERVICES AGREEMENT
This Agreement is entered into on January 15, 2024, by and between Alpha Corp ("Customer")
and Beta Solutions LLC ("Provider").
1. TERM AND TERMINATION: Either party may terminate this Agreement for convenience upon
providing thirty (30) days prior written notice to the other party.
2. LIABILITY: In no event shall either party's aggregate liability exceed one hundred thousand
dollars ($100,000), provided that neither party shall be subject to uncapped liability.
3. GOVERNING LAW: This Agreement shall be construed and governed in accordance with the laws
of the State of Delaware, without regard to conflict of laws principles.
4. CONFIDENTIALITY: Each party agrees to hold all proprietary information in strict confidence
for a period of five (5) years following termination.
5. NON-SOLICITATION: Provider agrees not to solicit or hire any employees of Customer for
twelve (12) months following contract completion.
"""


def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculates min, mean, median (p50), p90, p95, p99, and max from a list of latencies."""
    if not latencies_ms:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    s = sorted(latencies_ms)
    n = len(s)

    def p(pct: float) -> float:
        idx = max(0, min(int(n * pct) - 1, n - 1))
        return s[idx]

    return {
        "min": round(min(s), 2),
        "mean": round(statistics.mean(s), 2),
        "p50": round(statistics.median(s), 2),
        "p90": round(p(0.90), 2),
        "p95": round(p(0.95), 2),
        "p99": round(p(0.99), 2),
        "max": round(max(s), 2),
    }


async def benchmark_endpoint(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    payload_gen,
    total_requests: int,
    concurrency: int,
    task_name: str,
) -> Dict[str, Any]:
    """Runs a concurrent async load test on a specific endpoint."""
    print(f"\n[>>] Running Benchmark: [{task_name}]")
    print(f"     Requests: {total_requests} | Concurrency: {concurrency} workers | Target: {url}")

    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    status_codes: Dict[int, int] = {}
    errors: int = 0

    async def single_request(req_idx: int):
        nonlocal errors
        async with semaphore:
            payload = payload_gen(req_idx) if payload_gen else None
            t0 = time.perf_counter()
            try:
                if method == "GET":
                    resp = await client.get(url, timeout=30.0)
                elif method == "POST":
                    resp = await client.post(url, json=payload, timeout=30.0)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)
                status_codes[resp.status_code] = status_codes.get(resp.status_code, 0) + 1
                if resp.status_code >= 400:
                    errors += 1
            except Exception as e:
                errors += 1
                status_codes[500] = status_codes.get(500, 0) + 1

    overall_start = time.perf_counter()
    tasks = [asyncio.create_task(single_request(i)) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    overall_duration = time.perf_counter() - overall_start

    qps = round(total_requests / overall_duration, 2) if overall_duration > 0 else 0.0
    percentiles = compute_percentiles(latencies)
    success_rate = round(((total_requests - errors) / total_requests) * 100, 2) if total_requests > 0 else 0.0

    print(f"   [OK] Finished in {overall_duration:.2f}s | Throughput: {qps} req/s | Success Rate: {success_rate}%")
    print(f"   Latency: p50={percentiles['p50']}ms | p95={percentiles['p95']}ms | p99={percentiles['p99']}ms | max={percentiles['max']}ms")

    return {
        "benchmark": task_name,
        "endpoint": url,
        "method": method,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "total_duration_sec": round(overall_duration, 3),
        "requests_per_second": qps,
        "success_rate_pct": success_rate,
        "error_count": errors,
        "status_code_distribution": status_codes,
        "latency_percentiles_ms": percentiles,
    }


async def benchmark_upload_and_analysis(base_url: str, count: int = 3) -> Dict[str, Any]:
    """Benchmarks end-to-end contract upload, text extraction, and risk engine execution."""
    print(f"\n[>>] Running Benchmark: [End-to-End Contract Ingestion & Risk Scoring]")
    print(f"     Count: {count} contracts | Testing sync risk analysis pipeline")

    latencies: List[float] = []
    risk_scores: List[int] = []
    errors = 0

    async with httpx.AsyncClient(base_url=base_url) as client:
        start_overall = time.perf_counter()
        for i in range(count):
            t0 = time.perf_counter()
            try:
                # Upload contract text file
                files = {"file": (f"benchmark_contract_{i}.txt", SAMPLE_CONTRACT_TEXT.encode("utf-8"), "text/plain")}
                resp = await client.post("/api/contracts/upload?auto_analyze=false", files=files, timeout=30.0)
                if resp.status_code != 201:
                    errors += 1
                    continue
                cid = resp.json()["contract_id"]

                # Run synchronous analysis
                analysis_resp = await client.post(f"/api/contracts/{cid}/analyze?sync=true", timeout=60.0)
                if analysis_resp.status_code != 200:
                    errors += 1
                    continue

                duration_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(duration_ms)
                risk_scores.append(analysis_resp.json()["overall_risk_score"])
                print(f"   - Contract #{i+1} analyzed in {duration_ms:.1f}ms (Risk Score: {analysis_resp.json()['overall_risk_score']}/100)")
            except Exception as e:
                print(f"   - Contract #{i+1} failed: {e}")
                errors += 1

        total_time = time.perf_counter() - start_overall

    percentiles = compute_percentiles(latencies)
    qps = round(count / total_time, 2) if total_time > 0 else 0.0

    return {
        "benchmark": "End-to-End Ingestion & Risk Scoring",
        "endpoint": "/api/contracts/upload + /analyze",
        "method": "POST (Multipart + JSON)",
        "total_requests": count,
        "concurrency": 1,
        "total_duration_sec": round(total_time, 3),
        "requests_per_second": qps,
        "success_rate_pct": round(((count - errors) / count) * 100, 2) if count > 0 else 0.0,
        "error_count": errors,
        "latency_percentiles_ms": percentiles,
        "average_risk_score": round(statistics.mean(risk_scores), 1) if risk_scores else 0,
    }


async def run_all_benchmarks(base_url: str):
    """Executes the full test suite and persists benchmark findings."""
    print("=" * 70)
    print("ContractIQ - Production Load Testing & Concurrency Benchmark Suite")
    print(f"Target Base URL: {base_url}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    results = []

    limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)
    async with httpx.AsyncClient(base_url=base_url, limits=limits) as client:
        # 1. Healthcheck High-Concurrency Benchmark
        res_health = await benchmark_endpoint(
            client=client,
            method="GET",
            url="/health",
            payload_gen=None,
            total_requests=100,
            concurrency=10,
            task_name="Health Check High Concurrency",
        )
        results.append(res_health)

        # 2. Extended Health & System Readiness
        res_ext_health = await benchmark_endpoint(
            client=client,
            method="GET",
            url="/health/extended",
            payload_gen=None,
            total_requests=40,
            concurrency=5,
            task_name="Extended Health & Metrics Inspection",
        )
        results.append(res_ext_health)

        # 3. Dense FAISS Semantic Vector Search Concurrency
        def search_payload(idx: int):
            query = SEARCH_QUERIES[idx % len(SEARCH_QUERIES)]
            return {"query": query, "top_k": 5, "min_score": 0.15}

        res_search = await benchmark_endpoint(
            client=client,
            method="POST",
            url="/api/search/semantic",
            payload_gen=search_payload,
            total_requests=40,
            concurrency=5,
            task_name="FAISS Dense Vector Search",
        )
        results.append(res_search)

    # 4. End-to-End Contract Ingestion & Analysis
    res_e2e = await benchmark_upload_and_analysis(base_url=base_url, count=3)
    results.append(res_e2e)

    # Build final summary report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_url": base_url,
        "environment": "Production Test (Local ASGI / Uvicorn)",
        "benchmarks": results,
        "summary": {
            "total_benchmarks": len(results),
            "total_requests_executed": sum(r["total_requests"] for r in results),
            "overall_success_rate": round(statistics.mean(r["success_rate_pct"] for r in results), 2),
            "health_p95_latency_ms": res_health["latency_percentiles_ms"]["p95"],
            "vector_search_p95_latency_ms": res_search["latency_percentiles_ms"]["p95"],
            "vector_search_qps": res_search["requests_per_second"],
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY & PERFORMANCE RESULTS")
    print("=" * 70)
    for r in results:
        p = r["latency_percentiles_ms"]
        print(f"* {r['benchmark']:<38} | QPS: {r['requests_per_second']:>6.1f} | p50: {p['p50']:>6.1f}ms | p95: {p['p95']:>6.1f}ms | Success: {r['success_rate_pct']}%")
    print(f"\nDetailed benchmark report exported to:\n  {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ContractIQ load testing suite.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the FastAPI backend")
    args = parser.parse_args()

    asyncio.run(run_all_benchmarks(args.base_url))
