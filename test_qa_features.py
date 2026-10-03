"""Tests for QA label mapping, no-answer handling, and span decoding."""
from __future__ import annotations

import pytest

from ml.src.chunking import build_windows_for_clause, tokenize_context
from ml.src.qa_features import (
    answer_token_positions,
    build_eval_features,
    build_train_features,
    decode_text,
    decode_token_span,
)
from ml.src.cuad_loader import AnswerSpan, ClauseQA
from conftest import make_clause, synthetic_record


def _windows(tokenizer, config, record):
    tokenized = tokenize_context(tokenizer, record)
    clause = make_clause()
    q_ids = tokenizer(
        clause.question, add_special_tokens=False
    )["input_ids"]
    return build_windows_for_clause(
        tokenized, q_ids, "test_clause", config,
        tokenizer.cls_token_id, tokenizer.sep_token_id,
    )


def test_positive_answer_maps_to_correct_tokens(tokenizer, small_config):
    record = synthetic_record(120, (70, 75))
    windows = _windows(tokenizer, small_config, record)
    feats = build_train_features(windows, record.qas[0])
    positives = [f for f in feats if not f.is_no_answer]

    assert positives

    for f in positives:
        assert f.start_position <= f.end_position
        assert f.gold_char_span is not None
        w = windows[f.window_index]
        s = f.start_position - w.ctx_first_token_index_in_window
        e = f.end_position - w.ctx_first_token_index_in_window
        assert record.context[w.ctx_offsets[s][0]:w.ctx_offsets[e][1]] == \
            record.qas[0].answers[0].text


def test_no_answer_instance_maps_every_window_to_cls(tokenizer, small_config):
    record = synthetic_record(120, None)
    windows = _windows(tokenizer, small_config, record)
    feats = build_train_features(windows, record.qas[0])

    assert record.qas[0].is_impossible
    assert feats
    assert all(
        f.is_no_answer and f.start_position == 0 and f.end_position == 0
        and f.gold_char_span is None
        for f in feats
    )


def test_answer_outside_window_becomes_cls(tokenizer, small_config):
    record = synthetic_record(120, (2, 4))
    windows = _windows(tokenizer, small_config, record)

    for f in build_train_features(windows, record.qas[0]):
        w = windows[f.window_index]
        s, e = w.char_range()
        answer = record.qas[0].answers[0]

        if not (answer.start >= s and answer.end <= e):
            assert f.is_no_answer
            assert f.start_position == f.end_position == 0


def test_answer_crossing_boundary_is_never_clipped(tokenizer, small_config):
    record = synthetic_record(120, (30, 60))
    windows = _windows(tokenizer, small_config, record)
    gold = record.qas[0].answers[0]

    for w in windows:
        pos = answer_token_positions(w, gold)
        if pos:
            s, e = w.char_range()
            assert gold.start >= s and gold.end <= e

    for f in build_train_features(windows, record.qas[0]):
        if f.is_no_answer:
            assert f.start_position == f.end_position == 0


def test_multiple_gold_answers_all_get_supervision(tokenizer, small_config):
    record = synthetic_record(200, (20, 24))
    extra = AnswerSpan(
        text=record.context[300:309],
        start=300,
    )
    record.qas.append(
        ClauseQA(
            clause_label="test_clause",
            cuad_category="Test Category",
            question=record.qas[0].question,
            answers=[extra],
            is_impossible=False,
            qa_id="C__Test Category_1",
        )
    )

    windows = _windows(tokenizer, small_config, record)
    expected = {qa.answers[0].text for qa in record.qas}
    found = set()

    for qa in record.qas:
        for f in build_train_features(windows, qa):
            if not f.is_no_answer:
                w = windows[f.window_index]
                s = f.start_position - w.ctx_first_token_index_in_window
                e = f.end_position - w.ctx_first_token_index_in_window
                found.add(record.context[w.ctx_offsets[s][0]:w.ctx_offsets[e][1]])

    assert expected <= found


def test_eval_features_decode_prediction_back_to_text(tokenizer, small_config):
    record = synthetic_record(120, (70, 75))
    windows = _windows(tokenizer, small_config, record)
    features = build_eval_features(windows, record.qas[0])
    gold = record.qas[0].answers[0]

    for ef in features:
        w = windows[ef.window_index]
        pos = answer_token_positions(w, gold)
        if pos is not None:
            s, e = decode_token_span(
                ef, *pos, w.ctx_first_token_index_in_window
            )
            assert decode_text(record.context, s, e) == gold.text
            return

    pytest.fail("No window contained the gold answer")


def test_decode_rejects_out_of_context_span(tokenizer, small_config):
    record = synthetic_record(120, (70, 75))
    windows = _windows(tokenizer, small_config, record)
    features = build_eval_features(windows, record.qas[0])

    with pytest.raises(ValueError):
        decode_token_span(features[0], 1, 2, 10**9)


def test_answer_positions_none_outside(tokenizer, small_config):
    record = synthetic_record(120, (2, 4))
    windows = _windows(tokenizer, small_config, record)

    assert answer_token_positions(
        windows[-1], record.qas[0].answers[0]
    ) is None
