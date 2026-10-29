
#!/usr/bin/env python
"""Generate the ContractIQ clause registry from CUAD v1 question templates."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CUAD = PROJECT_ROOT / "data" / "raw" / "CUADv1.json"
OUTPUT_FILE = PROJECT_ROOT / "ml" / "configs" / "clauses.json"

CUAD_CATEGORY_PATTERN = re.compile(
    r'related to "(.+?)" that should be reviewed'
)

VERSION_1_CLAUSES = {
    "Document Name": "document_name",
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Renewal Term": "renewal_term",
    "Governing Law": "governing_law",
    "Termination For Convenience": "termination_for_convenience",
    "Non-Compete": "non_compete",
    "Exclusivity": "exclusivity",
    "Anti-Assignment": "anti_assignment",
    "License Grant": "license_grant",
    "Audit Rights": "audit_rights",
    "Cap On Liability": "cap_on_liability",
    "Insurance": "insurance",
}

INFERENCE_SETTINGS = {
    "max_seq_len": 512,
    "doc_stride": 128,
    "max_answer_tokens": 64,
    "no_answer_threshold": 0.35,
    "min_confidence": 0.25,
}

RISK_BANDS = {
    "low_max": 30,
    "medium_max": 60,
    "note": "0-30 LOW, 31-60 MEDIUM, 61-100 HIGH (configurable; heuristic engine only).",
}

EXPECTED_CATEGORY_COUNT = 41
GENERATED_DATE = "2026-08-30"


def load_cuad_dataset(path: Path) -> dict[str, Any]:
    """Load and validate the CUAD JSON dataset."""
    if not path.is_file():
        raise FileNotFoundError(
            f"CUAD dataset not found: {path}. Extract data.zip first."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in CUAD dataset: {exc}"
        ) from exc

    if not isinstance(dataset, dict) or not isinstance(
        dataset.get("data"), list
    ):
        raise ValueError(
            "Invalid CUAD dataset structure: expected a top-level 'data' list."
        )

    return dataset


def extract_question_templates(
    dataset: dict[str, Any],
) -> dict[str, str]:
    """Extract one original question template for each CUAD category."""
    templates: dict[str, str] = {}

    for document in dataset["data"]:
        paragraphs = document.get("paragraphs", [])

        if not paragraphs or not isinstance(paragraphs[0], dict):
            continue

        questions = paragraphs[0].get("qas", [])

        for question_data in questions:
            if not isinstance(question_data, dict):
                continue

            question_text = question_data.get("question", "")
            if not isinstance(question_text, str):
                continue

            match = CUAD_CATEGORY_PATTERN.search(question_text)
            if match:
                category = match.group(1)
                templates.setdefault(category, question_text)

    if len(templates) != EXPECTED_CATEGORY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORY_COUNT} CUAD categories, "
            f"but extracted {len(templates)}."
        )

    missing_categories = sorted(
        set(VERSION_1_CLAUSES) - set(templates)
    )

    if missing_categories:
        raise ValueError(
            f"Version-1 categories missing from CUAD: {missing_categories}"
        )

    return templates


def build_clause_registry(
    templates: dict[str, str],
) -> dict[str, Any]:
    """Build the complete ContractIQ configuration."""
    clauses = []

    for category in sorted(templates):
        enabled = category in VERSION_1_CLAUSES

        clauses.append({
            "label": VERSION_1_CLAUSES.get(category, ""),
            "cuad_category": category,
            "enabled": enabled,
            "question": templates[category],
        })

    enabled_count = sum(
        1 for clause in clauses if clause["enabled"]
    )

    return {
        "_meta": {
            "description": (
                "ContractIQ central clause registry. Generated from the "
                "actual CUAD question templates in data/raw/CUADv1.json "
                "(see scripts/generate_clause_config.py). To enable a new "
                "clause later, set \"enabled\": true and assign a stable "
                "\"label\"."
            ),
            "dataset": "CUAD v1 (Henriksson et al., 2021)",
            "generated_at": GENERATED_DATE,
            "n_categories_total": len(clauses),
            "n_enabled_v1": enabled_count,
            "disclaimer": (
                "AI-assisted analysis. Results should be reviewed by "
                "a qualified professional."
            ),
        },
        "inference": INFERENCE_SETTINGS.copy(),
        "risk_bands": RISK_BANDS.copy(),
        "clauses": clauses,
    }


def write_registry(
    configuration: dict[str, Any],
    destination: Path,
) -> None:
    """Create the output directory and write formatted JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    serialized_config = json.dumps(
        configuration,
        indent=2,
        ensure_ascii=False,
    )

    destination.write_text(
        serialized_config,
        encoding="utf-8",
    )


def main() -> int:
    """Run the clause registry generation workflow."""
    try:
        dataset = load_cuad_dataset(RAW_CUAD)
        templates = extract_question_templates(dataset)
        configuration = build_clause_registry(templates)
        write_registry(configuration, OUTPUT_FILE)

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metadata = configuration["_meta"]

    print(f"Generated clause registry: {OUTPUT_FILE}")
    print(
        f"Enabled clauses: {metadata['n_enabled_v1']} / "
        f"{metadata['n_categories_total']}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
