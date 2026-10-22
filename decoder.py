"""Decode QA logits into clause-level span decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowPrediction:
    window_index: int
    start_token: int
    end_token: int
    span_confidence: float
    cls_confidence: float


@dataclass(frozen=True)
class ClauseDecision:
    clause_label: str
    found: bool
    text: str
    confidence: float
    start_char: int
    end_char: int
    no_answer_score: float
    window_index: int | None
    windows_considered: int


def decode_window(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    ctx_from: int,
    ctx_to: int,
    max_answer_tokens: int,
    window_index: int = 0,
) -> WindowPrediction:
    """Return the best valid context span and CLS score."""

    start = torch.softmax(start_logits.float(), dim=-1)
    end = torch.softmax(end_logits.float(), dim=-1)
    cls_score = float(start[0] * end[0])

    n = ctx_to - ctx_from
    if n <= 0:
        pos = max(ctx_from, 0)
        return WindowPrediction(
            window_index, pos, pos, 0.0, cls_score
        )

    start_ctx = start[ctx_from:ctx_to]
    end_ctx = end[ctx_from:ctx_to]

    scores = start_ctx[:, None] * end_ctx[None, :]
    idx = torch.arange(n, device=scores.device)
    diff = idx[None, :] - idx[:, None]

    scores = scores.masked_fill(
        (diff < 0) | (diff >= max_answer_tokens),
        -1.0,
    )

    best = int(scores.flatten().argmax())
    start_idx, end_idx = divmod(best, n)

    return WindowPrediction(
        window_index=window_index,
        start_token=ctx_from + start_idx,
        end_token=ctx_from + end_idx,
        span_confidence=max(float(scores[start_idx, end_idx]), 0.0),
        cls_confidence=cls_score,
    )


def decide_clause(
    clause_label: str,
    predictions: list[WindowPrediction],
    context: str,
    start_char_offset: int,
    end_char_offset: int,
    no_answer_delta: float,
    min_confidence: float,
) -> ClauseDecision:
    """Combine window predictions into one clause decision."""

    if not predictions:
        return ClauseDecision(
            clause_label, False, "", 0.0, -1, -1,
            0.0, None, 0
        )

    best = max(predictions, key=lambda p: p.span_confidence)
    cls_score = max(p.cls_confidence for p in predictions)

    found = (
        best.span_confidence >= min_confidence
        and best.span_confidence - cls_score >= no_answer_delta
    )

    if not found:
        return ClauseDecision(
            clause_label=clause_label,
            found=False,
            text="",
            confidence=round(best.span_confidence, 6),
            start_char=-1,
            end_char=-1,
            no_answer_score=round(cls_score, 6),
            window_index=best.window_index,
            windows_considered=len(predictions),
        )

    if not (
        0 <= start_char_offset <= end_char_offset <= len(context)
    ):
        raise ValueError(
            "Decoded char offsets fall outside the original context."
        )

    return ClauseDecision(
        clause_label=clause_label,
        found=True,
        text=context[start_char_offset:end_char_offset],
        confidence=round(best.span_confidence, 6),
        start_char=start_char_offset,
        end_char=end_char_offset,
        no_answer_score=round(cls_score, 6),
        window_index=best.window_index,
        windows_considered=len(predictions),
    )


def decision_to_dict(decision: ClauseDecision) -> dict:
    return {
        "clause_type": decision.clause_label,
        "found": decision.found,
        "text": decision.text if decision.found else "",
        "confidence": decision.confidence,
        "start_char": decision.start_char,
        "end_char": decision.end_char,
        "no_answer_score": decision.no_answer_score,
        "windows_considered": decision.windows_considered,
    }
