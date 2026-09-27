"""ContractIQ - Celery Asynchronous Tasks for Distributed Processing.

Supports running contract analysis as asynchronous background worker jobs
when deployed with Redis/RabbitMQ message brokers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

# Broker configuration (defaults to local redis or env var)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery
    celery_app = Celery("contractiq", broker=REDIS_URL, backend=REDIS_URL)
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
except ImportError:
    celery_app = None


def run_contract_analysis(contract_id: str, text: str, title: str) -> Dict[str, Any]:
    """Execute contract intelligence and risk scoring."""
    from contract_risk_engine import ContractIntelligenceEngine

    engine = ContractIntelligenceEngine()
    result = engine.analyze_contract(text, title=title)

    return {
        "contract_id": contract_id,
        "contract_title": result.contract_title,
        "character_count": result.character_count,
        "word_count": result.word_count,
        "overall_risk_score": result.overall_risk_score,
        "overall_risk_level": result.overall_risk_level,
        "extracted_dates": result.extracted_dates,
        "durations": result.durations,
        "entities": result.entities,
        "clauses": [
            {
                "clause_type": c.clause_type,
                "display_name": c.display_name,
                "found": c.found,
                "confidence": c.confidence,
                "excerpt": c.excerpt,
                "risk_level": c.risk_level,
            }
            for c in result.clauses
        ],
        "risk_findings": [
            {
                "rule_id": rf.rule_id,
                "clause_type": rf.clause_type,
                "title": rf.title,
                "description": rf.description,
                "weight": rf.weight,
                "severity": rf.severity,
                "recommendation": rf.recommendation,
            }
            for rf in result.risk_findings
        ],
        "disclaimer": result.disclaimer,
    }


if celery_app is not None:
    @celery_app.task(name="tasks.analyze_contract_task")
    def analyze_contract_task(contract_id: str, text: str, title: str) -> Dict[str, Any]:
        return run_contract_analysis(contract_id, text, title)
