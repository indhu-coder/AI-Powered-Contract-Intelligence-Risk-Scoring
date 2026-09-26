"""ContractIQ - Week 2 Transformer Fine-Tuning Pipeline.

Fine-tunes a pre-trained transformer model on the CUAD dataset
to extract and classify legal contract clauses.

Usage:
    python train_clause_model.py --quick --epochs 1
    python train_clause_model.py --epochs 2 --batch-size 8 --model distilbert-base-uncased
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cuad-main" / "data" / "CUADv1.json"
OUTPUT_DIR = BASE_DIR / "models" / "fine_tuned_clause_model"
REPORTS_DIR = BASE_DIR / "reports"


class CUADQADataset(Dataset):
    """PyTorch Dataset for Extractive Legal QA features."""

    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


def extract_cuad_examples(limit_contracts: int | None = None) -> List[Dict]:
    """Parse raw CUAD JSON into flat QA training records."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    contracts = data.get("data", [])
    if limit_contracts:
        contracts = contracts[:limit_contracts]

    examples = []
    for item in contracts:
        title = item.get("title", "")
        for p in item.get("paragraphs", []):
            context = p.get("context", "")
            for qa in p.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", False)

                if not is_impossible and answers:
                    for ans in answers:
                        examples.append({
                            "title": title,
                            "context": context,
                            "question": question,
                            "answer_text": ans.get("text", ""),
                            "answer_start": ans.get("answer_start", 0),
                            "is_impossible": False,
                        })
                else:
                    examples.append({
                        "title": title,
                        "context": context,
                        "question": question,
                        "answer_text": "",
                        "answer_start": 0,
                        "is_impossible": True,
                    })

    return examples


def prepare_features(
    examples: List[Dict],
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    doc_stride: int = 128,
    positive_negative_ratio: float = 2.0,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Tokenize contexts and questions, mapping character spans to token indices."""
    import random
    rng = random.Random(seed)

    # Balance positive (answerable) and negative (no-answer) examples
    positives = [e for e in examples if not e["is_impossible"]]
    negatives = [e for e in examples if e["is_impossible"]]

    n_neg = int(len(positives) * positive_negative_ratio)
    sampled_neg = rng.sample(negatives, min(len(negatives), n_neg)) if n_neg < len(negatives) else negatives
    balanced_examples = positives + sampled_neg
    rng.shuffle(balanced_examples)

    questions = [e["question"] for e in balanced_examples]
    contexts = [e["context"] for e in balanced_examples]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_len,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i].item()
        example = balanced_examples[sample_idx]

        # No answer -> span points to CLS token (index 0)
        if example["is_impossible"] or not example["answer_text"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = example["answer_start"]
        end_char = start_char + len(example["answer_text"])
        sequence_ids = tokenized.sequence_ids(i)

        # Find context bounds in tokenized sequence
        ctx_start = 0
        while sequence_ids[ctx_start] != 1:
            ctx_start += 1
        ctx_end = len(sequence_ids) - 1
        while sequence_ids[ctx_end] != 1:
            ctx_end -= 1

        # Check if gold answer span falls entirely within this window
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Map start and end char offsets to token index
            token_start = ctx_start
            while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1

            token_end = ctx_end
            while token_end >= ctx_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            token_end += 1

            start_positions.append(token_start)
            end_positions.append(token_end)

    tokenized["start_positions"] = torch.tensor(start_positions)
    tokenized["end_positions"] = torch.tensor(end_positions)

    return tokenized


def train(
    model_name: str = "distilbert-base-uncased",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-5,
    quick: bool = False,
) -> Dict:
    """Execute the model fine-tuning loop."""
    print("=" * 60)
    print("CONTRACTIQ: TRANSFORMER FINE-TUNING PIPELINE")
    print(f"Base Model: {model_name}")
    print(f"Target Checkpoint: {OUTPUT_DIR}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Quick Mode: {quick}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    print("Loading Hugging Face tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print("Loading CUAD training dataset...")
    limit_contracts = 20 if quick else None
    examples = extract_cuad_examples(limit_contracts=limit_contracts)
    print(f"Extracted {len(examples)} raw QA instances from CUAD.")

    if quick:
        examples = examples[:250]
        print(f"Quick mode active: using {len(examples)} QA instances.")

    print("Tokenizing and preparing span features...")
    encodings = prepare_features(examples, tokenizer, max_len=384, doc_stride=128)
    n_samples = encodings["input_ids"].shape[0]
    print(f"Generated {n_samples} sliding window training features.")

    # Train / Val split (85% train, 15% validation)
    val_size = max(1, int(n_samples * 0.15))
    train_size = n_samples - val_size

    train_encodings = {k: v[:train_size] for k, v in encodings.items()}
    val_encodings = {k: v[train_size:] for k, v in encodings.items()}

    train_dataset = CUADQADataset(train_encodings)
    val_dataset = CUADQADataset(val_encodings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training windows: {train_size} | Validation windows: {val_size}")
    print("Loading pre-trained QA model weights...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    training_stats = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        step_count = 0

        print(f"\n--- Epoch {epoch} / {epochs} ---")
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            step_count += 1

            if (step + 1) % max(1, len(train_loader) // 5) == 0 or (step + 1) == len(train_loader):
                avg_step_loss = total_train_loss / step_count
                print(f"  Step {step + 1}/{len(train_loader)} | Training Loss: {avg_step_loss:.4f}")

        avg_train_loss = total_train_loss / max(1, step_count)

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_pos = batch["start_positions"].to(device)
                end_pos = batch["end_positions"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                )
                total_val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        print(f"Epoch {epoch} Complete | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        training_stats.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        })

    elapsed_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 60)
    print(f"Training completed in {elapsed_time}s.")

    # Save fine-tuned checkpoint & tokenizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned model checkpoint to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save training report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "week2_training_report.json"
    report = {
        "model_name": model_name,
        "device": device.type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_time_seconds": elapsed_time,
        "total_windows": n_samples,
        "train_windows": train_size,
        "val_windows": val_size,
        "history": training_stats,
        "checkpoint_dir": str(OUTPUT_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote training report to {report_file}")
    print("=" * 60)
    return report


def main():
    parser = argparse.ArgumentParser(description="ContractIQ Week 2 Transformer Fine-Tuning")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Base HF model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Run fast verification sample")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
