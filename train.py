"""ContractIQ extractive QA training pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import PROCESSED_DIR
from .device import get_device

DEFAULT_BASELINE = "deepset/minilm-uncased-squad2"
TRAIN_DIR = PROCESSED_DIR / "train_features"


class WindowDataset(Dataset):
    """Dataset wrapper for sampled QA feature rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def load_train_table(limit=None):
    """Load Phase 2 Parquet features into memory."""
    import pyarrow.parquet as pq

    files = sorted(TRAIN_DIR.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No training features under {TRAIN_DIR}. "
            "Run scripts/prepare_phase2_data.py first."
        )

    rows = []

    for path in files:
        table = pq.read_table(path)
        data = {
            col: table.column(col).to_pylist()
            for col in table.column_names
        }

        for i in range(table.num_rows):
            rows.append({col: data[col][i] for col in data})

            if limit and len(rows) >= limit:
                return rows

    return rows


def sample_train_rows(
    rows,
    negative_ratio=3.0,
    sample_limit=None,
    seed=42,
):
    """Keep positives and a reproducible sample of negative windows."""
    rng = random.Random(seed)

    positives = [r for r in rows if not r["is_no_answer"]]
    negatives = [r for r in rows if r["is_no_answer"]]

    count = min(
        len(negatives),
        int(round(len(positives) * negative_ratio)),
    )

    selected_negatives = (
        rng.sample(negatives, count)
        if count < len(negatives)
        else list(negatives)
    )

    sampled = positives + selected_negatives
    rng.shuffle(sampled)

    stats = {
        "seed": seed,
        "negative_ratio": negative_ratio,
        "positives_available": len(positives),
        "negatives_available": len(negatives),
        "positives_kept": len(positives),
        "negatives_kept": len(selected_negatives),
        "total_rows": len(sampled),
        "shuffled": True,
    }

    if sample_limit and len(sampled) > sample_limit:
        sampled = sampled[:sample_limit]
        stats["sample_limit"] = sample_limit
        stats["total_rows_after_limit"] = len(sampled)

    return sampled, stats


def collate(batch, pad_id):
    """Pad sequences and convert labels to tensors."""
    max_len = max(len(row["input_ids"]) for row in batch)

    input_ids = torch.full(
        (len(batch), max_len),
        pad_id,
        dtype=torch.long,
    )
    attention = torch.zeros(
        (len(batch), max_len),
        dtype=torch.long,
    )

    for i, row in enumerate(batch):
        ids = torch.tensor(row["input_ids"], dtype=torch.long)
        input_ids[i, :len(ids)] = ids
        attention[i, :len(ids)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "start_positions": torch.tensor(
            [r["start_position"] for r in batch],
            dtype=torch.long,
        ),
        "end_positions": torch.tensor(
            [r["end_position"] for r in batch],
            dtype=torch.long,
        ),
    }


def evaluate_quick(model, rows, pad_id, device, batch_size=16):
    """Calculate sampled token start/end accuracy."""
    model.eval()
    start_correct = end_correct = total = 0

    loader = DataLoader(
        WindowDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_id),
    )

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            pred_start = output.start_logits.argmax(dim=-1)
            pred_end = output.end_logits.argmax(dim=-1)

            valid = batch["start_positions"] >= 0

            start_correct += (
                pred_start[valid] ==
                batch["start_positions"][valid]
            ).sum().item()

            end_correct += (
                pred_end[valid] ==
                batch["end_positions"][valid]
            ).sum().item()

            total += int(valid.sum().item())

    model.train()

    return {
        "token_start_accuracy": (
            start_correct / total if total else 0.0
        ),
        "token_end_accuracy": (
            end_correct / total if total else 0.0
        ),
        "rows": total,
    }


def save_checkpoint(model, tokenizer, directory, metadata):
    """Save model, tokenizer, and training metadata."""
    directory.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

    (directory / "training_meta.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def train(args):
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = get_device("cpu" if args.cpu else None)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    pin_memory = device.type == "cuda"

    print(f"[train] device={device} seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model
    ).to(device)

    pad_id = tokenizer.pad_token_id

    print("[train] loading Phase 2 train features ...")
    started = time.perf_counter()

    all_rows = load_train_table(args.read_limit)

    rows, sampling = sample_train_rows(
        all_rows,
        negative_ratio=args.negative_ratio,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )

    del all_rows

    print(
        f"[train] features ready in "
        f"{time.perf_counter() - started:.0f}s | "
        f"positives {sampling['positives_kept']}, "
        f"negatives {sampling['negatives_kept']}, "
        f"total {sampling['total_rows']}"
    )

    rng = random.Random(args.seed + 1)
    val_rows = rng.sample(
        rows,
        min(args.val_rows, len(rows)),
    )

    loader = DataLoader(
        WindowDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, pad_id),
        num_workers=0,
        drop_last=False,
        pin_memory=pin_memory,
    )

    steps_per_epoch = math.ceil(
        len(loader) / args.grad_accum
    )
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and args.amp,
    )

    print(
        f"[train] {len(rows)} rows | "
        f"{steps_per_epoch} steps/epoch | "
        f"{total_steps} total steps | "
        f"batch {args.batch_size} x accum {args.grad_accum}"
    )

    best_score = -1.0
    history = []
    global_step = 0
    started = time.perf_counter()
    stop = False

    model.train()

    for epoch in range(1, args.epochs + 1):
        if stop:
            break

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, 1):
            batch = {
                key: value.to(
                    device,
                    non_blocking=pin_memory,
                )
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and args.amp,
            ):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                loss = output.loss / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += output.loss.item()

                if global_step % args.log_every == 0:
                    print(
                        f"[train] epoch {epoch} "
                        f"step {global_step}/{total_steps} "
                        f"loss {running_loss / args.log_every:.4f} "
                        f"({time.perf_counter() - started:.0f}s)",
                        flush=True,
                    )
                    running_loss = 0.0

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                print(
                    f"[train] --max-steps "
                    f"{args.max_steps} reached; stopping"
                )
                break

            if (
                args.time_limit_min
                and time.perf_counter() - started
                > args.time_limit_min * 60
            ):
                stop = True
                print(
                    f"[train] --time-limit "
                    f"{args.time_limit_min}min reached; stopping"
                )
                break

        metrics = evaluate_quick(
            model,
            val_rows,
            pad_id,
            device,
        )

        metrics.update({
            "epoch": epoch,
            "global_step": global_step,
            "wall_seconds": round(
                time.perf_counter() - started,
                1,
            ),
        })

        history.append(metrics)

        print(
            f"[train] epoch {epoch} "
            f"quick metrics: {metrics}"
        )

        metadata = {
            "args": vars(args),
            "sampling": sampling,
            "history": history,
            "device": device,
            "saved_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_checkpoint(
            model,
            tokenizer,
            Path(args.checkpoints) / f"epoch{epoch}",
            metadata,
        )

        score = (
            metrics["token_start_accuracy"]
            + metrics["token_end_accuracy"]
        )

        if score > best_score:
            best_score = score

            save_checkpoint(
                model,
                tokenizer,
                Path(args.output),
                {
                    **metadata,
                    "best": True,
                },
            )

            print(
                f"[train] saved best model "
                f"to {args.output}"
            )

    print(
        f"[train] done in "
        f"{time.perf_counter() - started:.0f}s | "
        f"best score {best_score:.4f}"
    )

    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="ContractIQ QA fine-tuning"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_BASELINE,
        help="HF model id or path",
    )
    parser.add_argument(
        "--train-dir",
        default=str(TRAIN_DIR),
    )
    parser.add_argument(
        "--output",
        default="ml/models/final",
    )
    parser.add_argument(
        "--checkpoints",
        default="ml/checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=3e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=3.0,
        help="no-answer windows per positive window",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--read-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--val-rows",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="mixed precision on CUDA",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--time-limit-min",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="run a small CPU proof test",
    )

    return parser


def main():
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[2]

    args.output = str(
        (root / args.output).resolve()
    )
    args.checkpoints = str(
        (root / args.checkpoints).resolve()
    )

    if args.tiny:
        args.epochs = 1
        args.batch_size = 8
        args.sample_limit = args.sample_limit or 2048
        args.read_limit = args.read_limit or 120_000
        args.max_steps = args.max_steps or 120
        args.time_limit_min = args.time_limit_min or 15
        args.cpu = True

        args.output = str(
            (root / "ml/models/tiny_cpu").resolve()
        )
        args.checkpoints = str(
            (root / "ml/checkpoints/tiny_cpu").resolve()
        )

    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
