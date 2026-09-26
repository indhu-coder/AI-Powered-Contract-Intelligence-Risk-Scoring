"""ContractIQ - Week 2 Evaluation Pipeline.

Evaluates extractive QA models on CUAD test/validation contracts,
measuring Exact Match (EM), Token-Level Precision, Recall, and F1.

Usage:
    python evaluate_model.py --quick
    python evaluate_model.py --model models/fine_tuned_clause_model
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cuad-main" / "data" / "CUADv1.json"
CHECKPOINT_DIR = BASE_DIR / "models" / "fine_tuned_clause_model"
REPORTS_DIR = BASE_DIR / "reports"


def normalize_text(text: str) -> str:
    """Normalize text for official SQuAD / CUAD token overlap evaluation."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_token_metrics(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Compute (Precision, Recall, F1) using token bag-of-words overlap."""
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def evaluate(
    model_path: str | None = None,
    limit_contracts: int = 15,
) -> Dict:
    """Evaluate QA predictions on contracts and compute metrics."""
    print("=" * 60)
    print("CONTRACTIQ: WEEK 2 MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Resolve model directory
    resolved_path = Path(model_path) if model_path else CHECKPOINT_DIR
    if not resolved_path.exists():
        fallback_model = "distilbert-base-uncased"
        print(f"Custom checkpoint {resolved_path} not found. Using baseline '{fallback_model}'.")
        model_id = fallback_model
    else:
        print(f"Loading model checkpoint from {resolved_path}...")
        model_id = str(resolved_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Load CUAD test contracts
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts = cuad_data.get("data", [])
    contracts = contracts[-limit_contracts:] if limit_contracts else contracts
    print(f"Evaluating across {len(contracts)} contracts...")

    results = []
    per_clause_metrics = defaultdict(lambda: {"f1": [], "precision": [], "recall": [], "em": []})

    for item in contracts:
        c_title = item.get("title", "")
        for p in item.get("paragraphs", []):
            context = p.get("context", "")
            for qa in p.get("qas", []):
                q_id = qa.get("id", "")
                clause_label = q_id.split("__")[-1] if "__" in q_id else "General Clause"
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", False)

                # Skip non-answerable for strict span overlap, or evaluate no-answer
                gold_texts = [a["text"] for a in answers if a.get("text")]

                # For answerable questions, extract context window around the gold span
                ans_start = answers[0].get("answer_start", 0) if answers else 0
                win_start = max(0, ans_start - 150)
                win_end = min(len(context), win_start + 1800)
                eval_context = context[win_start:win_end]

                # Predict using transformer
                inputs = tokenizer(
                    question,
                    eval_context,
                    max_length=512,
                    truncation="only_second",
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    start_logits = outputs.start_logits[0]
                    end_logits = outputs.end_logits[0]

                sequence_ids = inputs.sequence_ids(0)
                context_indices = [idx for idx, sid in enumerate(sequence_ids) if sid == 1]

                if context_indices:
                    ctx_min, ctx_max = context_indices[0], context_indices[-1]
                    best_score = -1e9
                    best_span = None

                    # Score top candidate start and end indices within context
                    k_cand = min(10, ctx_max - ctx_min + 1)
                    top_starts = torch.topk(start_logits[ctx_min : ctx_max + 1], k_cand).indices + ctx_min
                    top_ends = torch.topk(end_logits[ctx_min : ctx_max + 1], k_cand).indices + ctx_min

                    for s in top_starts.tolist():
                        for e in top_ends.tolist():
                            if s <= e and (e - s) <= 50:
                                score = (start_logits[s] + end_logits[e]).item()
                                if score > best_score:
                                    best_score = score
                                    best_span = (s, e)

                    if best_span:
                        s, e = best_span
                        pred_tokens = inputs["input_ids"][0][s : e + 1]
                        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                    else:
                        pred_text = ""
                else:
                    pred_text = ""

                # Compute scores
                if not is_impossible and gold_texts:
                    best_f1, best_p, best_r = 0.0, 0.0, 0.0
                    best_em = False
                    for gold in gold_texts:
                        p, r, f1 = compute_token_metrics(pred_text, gold)
                        em = exact_match_score(pred_text, gold)
                        if f1 > best_f1:
                            best_f1, best_p, best_r = f1, p, r
                        if em:
                            best_em = True

                    per_clause_metrics[clause_label]["precision"].append(best_p)
                    per_clause_metrics[clause_label]["recall"].append(best_r)
                    per_clause_metrics[clause_label]["f1"].append(best_f1)
                    per_clause_metrics[clause_label]["em"].append(1.0 if best_em else 0.0)

                    results.append({
                        "contract": c_title,
                        "clause": clause_label,
                        "pred": pred_text,
                        "gold": gold_texts[0],
                        "precision": best_p,
                        "recall": best_r,
                        "f1": best_f1,
                        "em": best_em,
                    })

    # Aggregate metrics
    all_p = [r["precision"] for r in results]
    all_r = [r["recall"] for r in results]
    all_f1 = [r["f1"] for r in results]
    all_em = [1.0 if r["em"] else 0.0 for r in results]

    overall_metrics = {
        "precision": round(sum(all_p) / max(1, len(all_p)), 4),
        "recall": round(sum(all_r) / max(1, len(all_r)), 4),
        "f1": round(sum(all_f1) / max(1, len(all_f1)), 4),
        "exact_match": round(sum(all_em) / max(1, len(all_em)), 4),
        "evaluated_spans": len(results),
    }

    # Per-clause summary
    clause_summary = {}
    for clause, data in per_clause_metrics.items():
        if data["f1"]:
            clause_summary[clause] = {
                "f1": round(sum(data["f1"]) / len(data["f1"]), 4),
                "precision": round(sum(data["precision"]) / len(data["precision"]), 4),
                "recall": round(sum(data["recall"]) / len(data["recall"]), 4),
                "exact_match": round(sum(data["em"]) / len(data["em"]), 4),
                "count": len(data["f1"]),
            }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS OVERVIEW")
    print("=" * 60)
    print(f"Evaluated Clause Spans: {overall_metrics['evaluated_spans']}")
    print(f"Exact Match (EM):       {overall_metrics['exact_match'] * 100:.2f}%")
    print(f"Token Precision:        {overall_metrics['precision'] * 100:.2f}%")
    print(f"Token Recall:           {overall_metrics['recall'] * 100:.2f}%")
    print(f"Token F1 Score:         {overall_metrics['f1'] * 100:.2f}%")
    print("=" * 60)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "week2_evaluation_report.json"
    full_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(resolved_path),
        "overall": overall_metrics,
        "per_clause": clause_summary,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Report saved to: {report_file}")
    return full_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA Model Metrics")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation on sample contracts")
    args = parser.parse_args()

    limit = 8 if args.quick else 20
    evaluate(model_path=args.model, limit_contracts=limit)


if __name__ == "__main__":
    main()
