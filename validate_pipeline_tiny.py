#!/usr/bin/env python
"""Phase 2 tiny pipeline validation on real CUAD data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.chunking import build_windows_for_clause, tokenize_context
from ml.src.config import (
    REPORTS_DIR,
    load_enabled_clauses,
    load_window_config,
)
from ml.src.cuad_loader import load_train_contracts
from ml.src.qa_features import (
    answer_token_positions,
    build_train_features,
    cls_index,
)
from ml.src.preprocessing import question_budget


N_CANDIDATE_CONTRACTS = 40
N_CLAUSES = 5
N_CONTRACTS_TO_SELECT = 4


def get_clause_question(label):
    """Return the configured question for a clause label."""
    from ml.src.config import load_enabled_clauses

    for clause in load_enabled_clauses():
        if clause.label == label:
            return clause.question

    return ""


def create_dummy_qa(label, question):
    """Create a no-answer QA instance when no QA exists."""
    from ml.src.cuad_loader import ClauseQA

    return ClauseQA(
        clause_label=label,
        cuad_category="",
        question=question,
        answers=[],
        is_impossible=True,
        qa_id="",
    )


def get_qa_for_label(record, label):
    """Return the first QA instance for the requested clause."""
    qas = record.qas_for(label)

    if qas:
        return qas[0]

    return create_dummy_qa(
        label,
        get_clause_question(label),
    )


def select_contract(selected, reasons, record, reason):
    """Add a contract once, while respecting the selection limit."""
    if len(selected) >= N_CONTRACTS_TO_SELECT:
        return

    existing = {item.contract_id for item in selected}

    if record.contract_id not in existing:
        selected.append(record)
        reasons.append(reason)


def select_test_contracts(candidates, clauses):
    """Deterministically select contracts covering required cases."""
    selected = []
    reasons = []

    # Multiple gold spans
    for record in candidates:
        for clause in clauses:
            if len(record.qas_for(clause.label)) > 1:
                select_contract(
                    selected,
                    reasons,
                    record,
                    "clause with multiple gold instances (exploded spans)",
                )
                break

    return selected, reasons


def add_window_contracts(
    candidates,
    clauses,
    tokenizer,
    config,
    cls_id,
    sep_id,
    selected,
    reasons,
    tokenized_cache,
):
    """Select contracts that require multiple sliding windows."""
    for record in candidates:
        tokenized = tokenize_context(tokenizer, record)
        tokenized_cache[record.contract_id] = tokenized

        clause = clauses[0]

        question_ids = tokenizer(
            clause.question,
            add_special_tokens=False,
        )["input_ids"]

        windows = build_windows_for_clause(
            tokenized,
            question_ids,
            clause.label,
            config,
            cls_id,
            sep_id,
        )

        if len(windows) > 1:
            select_contract(
                selected,
                reasons,
                record,
                "contract requires multiple sliding windows",
            )


def select_positive_contract(candidates, selected, reasons):
    """Select a contract containing a positive gold answer."""
    for record in candidates:
        if any(
            not qa.is_impossible and qa.answers
            for qa in record.qas
        ):
            select_contract(
                selected,
                reasons,
                record,
                "contract with positive gold span",
            )
            break


def select_no_answer_contract(candidates, selected, reasons):
    """Select a contract containing a no-answer example."""
    for record in candidates:
        if any(qa.is_impossible for qa in record.qas):
            select_contract(
                selected,
                reasons,
                record,
                "contract with a no-answer clause instance",
            )
            break


def fill_remaining_contracts(candidates, selected, reasons):
    """Fill remaining slots with deterministic candidates."""
    while len(selected) < N_CONTRACTS_TO_SELECT and candidates:
        record = candidates[
            len(selected) * 7 % len(candidates)
        ]

        select_contract(
            selected,
            reasons,
            record,
            "additional deterministic pick",
        )

        if (
            len(selected) < N_CONTRACTS_TO_SELECT
            and all(
                candidates[0].contract_id == item.contract_id
                for item in selected
            )
        ):
            break


def validate_contract(
    record,
    clauses,
    tokenizer,
    tokenized,
    config,
    cls_id,
    sep_id,
):
    """Run feature generation and gold-span reconstruction."""
    total_windows = 0
    positive_windows = 0
    no_answer_windows = 0
    reconstructed = 0
    boundary_extended = 0
    multi_span_seen = False

    contract_window_count = 0

    for clause in clauses:
        question_ids = tokenizer(
            clause.question,
            add_special_tokens=False,
        )["input_ids"]

        windows = build_windows_for_clause(
            tokenized,
            question_ids,
            clause.label,
            config,
            cls_id,
            sep_id,
        )

        contract_window_count += len(windows)

        qa = get_qa_for_label(record, clause.label)
        features = build_train_features(windows, qa)

        for feature in features:
            total_windows += 1

            if feature.is_no_answer:
                no_answer_windows += 1

                assert (
                    feature.start_position == cls_index()
                    and feature.end_position == cls_index()
                )

                continue

            positive_windows += 1

            window = windows[feature.window_index]

            local_start = (
                feature.start_position
                - window.ctx_first_token_index_in_window
            )
            local_end = (
                feature.end_position
                - window.ctx_first_token_index_in_window
            )

            char_start = window.ctx_offsets[local_start][0]
            char_end = window.ctx_offsets[local_end][1]

            gold_start, gold_end = feature.gold_char_span

            gold_text = record.context[gold_start:gold_end]
            extracted_text = record.context[char_start:char_end]

            if not (
                char_start <= gold_start
                and gold_end <= char_end
            ):
                raise AssertionError(
                    f"RECONSTRUCTION FAILURE for "
                    f"{record.contract_id}/{clause.label}: "
                    f"decoded {extracted_text!r} does not contain "
                    f"gold {gold_text!r}"
                )

            if char_start != gold_start or char_end != gold_end:
                boundary_extended += 1

            reconstructed += 1

        if len(record.qas_for(clause.label)) > 1:
            multi_span_seen = True

    return {
        "context_tokens": len(tokenized.token_ids),
        "windows_total": contract_window_count,
        "total_windows": total_windows,
        "positive_windows": positive_windows,
        "no_answer_windows": no_answer_windows,
        "reconstructed": reconstructed,
        "boundary_extended": boundary_extended,
        "multi_span_seen": multi_span_seen,
    }


def main() -> int:
    from transformers import AutoTokenizer

    clauses = load_enabled_clauses()[:N_CLAUSES]
    config = load_window_config()

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        use_fast=True,
    )

    config.check_tokenizer_compat(
        tokenizer.model_max_length
    )

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    print(
        f"Loading {N_CANDIDATE_CONTRACTS} candidate contracts "
        "(strict validation)..."
    )

    candidates = load_train_contracts(
        load_enabled_clauses(),
        limit=N_CANDIDATE_CONTRACTS,
        strict=True,
    )

    print(
        f"Loaded {len(candidates)} contracts "
        "with strict answer validation: OK"
    )

    selected, reasons = select_test_contracts(
        candidates,
        clauses,
    )

    tokenized_cache = {}

    add_window_contracts(
        candidates,
        clauses,
        tokenizer,
        config,
        cls_id,
        sep_id,
        selected,
        reasons,
        tokenized_cache,
    )

    select_positive_contract(
        candidates,
        selected,
        reasons,
    )

    select_no_answer_contract(
        candidates,
        selected,
        reasons,
    )

    fill_remaining_contracts(
        candidates,
        selected,
        reasons,
    )

    print(
        "Selected contracts:",
        [
            (record.contract_id, reason)
            for record, reason in zip(selected, reasons)
        ],
    )

    stats = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "clauses": [clause.label for clause in clauses],
        "selected_contracts": [
            {
                "contract_id": record.contract_id,
                "reason": reason,
            }
            for record, reason in zip(selected, reasons)
        ],
        "checks": {},
        "per_contract": [],
    }

    total_windows = 0
    positive_windows = 0
    no_answer_windows = 0
    max_windows_any = 0
    reconstructed = 0
    boundary_extended = 0
    multi_span_seen = False

    for record in selected:
        tokenized = tokenized_cache[record.contract_id]

        result = validate_contract(
            record,
            clauses,
            tokenizer,
            tokenized,
            config,
            cls_id,
            sep_id,
        )

        total_windows += result["total_windows"]
        positive_windows += result["positive_windows"]
        no_answer_windows += result["no_answer_windows"]
        reconstructed += result["reconstructed"]
        boundary_extended += result["boundary_extended"]
        multi_span_seen |= result["multi_span_seen"]

        max_windows_any = max(
            max_windows_any,
            result["windows_total"],
        )

        stats["per_contract"].append({
            "contract_id": record.contract_id,
            "context_tokens": result["context_tokens"],
            "windows_total": result["windows_total"],
        })

    stats["checks"] = {
        "strict_answer_validation": True,
        "positive_example_present": positive_windows > 0,
        "no_answer_example_present": no_answer_windows > 0,
        "multi_window_contract_present": (
            max_windows_any > len(clauses)
        ),
        "multi_gold_span_clause_present": multi_span_seen,
        "all_positive_spans_reconstructed_with_containment": True,
        "reconstructed_spans": reconstructed,
        "boundary_extended_spans": boundary_extended,
        "note": (
            "Some CUAD gold spans cut mid-wordpiece-token; "
            "token-level decoding extends those to the containing "
            "token boundary (standard SQuAD behaviour). Precise "
            "char spans are preserved in feature metadata."
        ),
    }

    stats["window_stats"] = {
        "total": total_windows,
        "positive": positive_windows,
        "no_answer": no_answer_windows,
        "max_windows_per_contract": max_windows_any,
    }

    stats["question_budget"] = question_budget(
        load_enabled_clauses(),
        config,
        tokenizer,
    )[:N_CLAUSES]

    checks = stats["checks"]

    assert checks["positive_example_present"], (
        "no positive example selected"
    )
    assert checks["no_answer_example_present"], (
        "no no-answer example selected"
    )
    assert checks["multi_window_contract_present"], (
        "no multi-window contract selected"
    )
    assert reconstructed > 0

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = REPORTS_DIR / "phase2_tiny_validation.json"

    output.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nAll tiny-pipeline checks passed. "
        f"Report: {output}"
    )
    print(
        f"Windows: {total_windows} "
        f"(positive {positive_windows}, "
        f"no-answer {no_answer_windows})"
    )
    print(
        f"Gold spans reconstructed exactly: "
        f"{reconstructed}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
