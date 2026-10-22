"""ContractIQ ML inference service."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
import re
import torch

from chunking import (
    TokenizedContext,
    build_windows_for_token_slice,
    tokenize_context,
    validate_window_fit,
)
from config import (
    ML_CONFIG_DIR,
    PROJECT_ROOT,
    load_enabled_clauses,
    load_window_config,
)
from decoder import ClauseDecision, WindowPrediction, decode_window, decision_to_dict
from device import device_summary, get_device
from retrieval import merge_regions, select_regions


MODEL_CONFIG_PATH = ML_CONFIG_DIR / "model.json"
FINAL_MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "final"


MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 32

class InferenceError(RuntimeError):
    """Raised when no usable QA model can be loaded."""


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict:
    if not path.exists():
        raise InferenceError(f"Model config not found at {path}. The ML layer is not set up.")
    return json.loads(path.read_text(encoding="utf-8"))
def split_sections(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections = []
    for paragraph in paragraphs:
        if len(paragraph) <= 4000:
            sections.append(paragraph)
        else:
            # Overlapping windows preserve context in long paragraphs.
            step = 3000
            sections.extend(paragraph[i:i + 3500].strip() for i in range(0, len(paragraph), step))
    return [s for s in sections if s] or [text]
def split_sentences(text: str) -> list[str]:
    """Split contract text into sentences while preserving date formats."""
    text = re.sub(r"\s+", " ", text).strip()

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text,
        )
        if sentence.strip()
    ]

STOPWORDS = {
    "what", "when", "where", "who", "which", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "for", "in", "on", "does", "do", "did", "this",
    "that", "under", "with", "and", "or", "by", "be", "it", "their", "its",
    "agreement", "please", "tell", "me"
}


def retrieve_sections(question, sections, limit=10):
    q = question.lower()
    terms = set(re.findall(r"[a-z]{3,}", q)) - STOPWORDS
    if any(x in q for x in ("notice", "terminate", "termination", "cancellation")):
        terms.update({"notice", "termination", "terminate", "terminated", "cancel"})
    if "effective date" in q or "commencement date" in q:
        terms.update({"effective", "commencement", "date", "dated"})
    if any(x in q for x in ("expiration date", "expiry date", "end date")):
        terms.update({"expiration", "expiry", "expires", "expire", "ends"})
    if "renewal term" in q:
        terms.update({"renewal", "renew", "term"})

    ranked = []
    for idx, section in enumerate(sections):
        low = section.lower()
        score = 2 * sum(bool(re.search(rf"\b{re.escape(t)}\b", low)) for t in terms if t not in {"terminate", "cancel"})
        if re.search(r"\b(terminat\w*|cancel\w*)\b", low) and any(x in q for x in ("terminate", "termination", "cancellation")):
            score += 6
        if "notice" in q and re.search(r"\bnotice\b", low):
            score += 8
        if any(x in q for x in ("notice", "terminate", "termination")) and re.search(r"\b(days?|weeks?|months?|years?)\b", low):
            score += 2
        if "effective date" in q and "effective" in low:
            score += 10
        if any(x in q for x in ("expiration date", "expiry date", "end date")) and re.search(r"\b(expir\w*|ends on|ending on)\b", low):
            score += 8
        ranked.append((score, idx, section))

    selected = sorted((x for x in ranked if x[0] > 0), key=lambda x: (-x[0], x[1]))[:limit]
    if not selected:
        return sections[:limit]
    selected.sort(key=lambda x: x[1])
    return [x[2] for x in selected]
def targeted_duration_answer(question, context):
    q = question.lower()
    if not any(x in q for x in ("notice", "terminate", "termination", "renewal term", "how long")):
        return None
    candidates = []
    for sentence in split_sentences(context):
        low = sentence.lower()
        if not re.search(r"\b(notice|terminat\w*|cancel\w*|renewal)\b", low):
            continue
        match = DURATION_RE.search(sentence)
        if not match:
            continue
        if "notice" in q and "notice" not in low:
            continue
        # Return the relevant contractual sentence, preserving words such as "prior notice".
        candidates.append((sentence, bool(re.search(r"\bnotice\b", low)) + bool(re.search(r"\bterminat\w*\b", low))))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    sentence = candidates[0][0]
    return {"answer": sentence, "evidence": sentence, "source": "Relevant notice/termination clause"}


def targeted_date_answer(question, context):
    q = question.lower()
    if not any(x in q for x in ("effective date", "commencement date", "expiration date", "expiry date", "agreement date", "execution date")):
        return None
    if "effective" in q or "commencement" in q:
        key = r"\b(effective date|commencement date|becomes effective|shall be effective)\b"
    elif "expiration" in q or "expiry" in q:
        key = r"\b(expiration date|expiry date|expires|shall expire|ends on|ending on)\b"
    else:
        key = r"\b(agreement date|dated as of|executed on|execution date)\b"
    date_re = re.compile(
        r"\b(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b", re.I
    )
    for sentence in split_sentences(context):
        if re.search(key, sentence, re.I):
            match = date_re.search(sentence)
            if match:
                return {"answer": match.group(0), "evidence": sentence, "source": "Matching date clause"}
    # Never return an unrelated date as a fallback.
    return None
def bert_answer(question, context, tokenizer, model, device):
    encoded = tokenizer(
        question, context, truncation="only_second", max_length=MAX_LENGTH,
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length", return_tensors="pt"
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    seq_ids = [encoded.sequence_ids(i) for i in range(offsets.shape[0])]
    model_inputs = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs)

    best = None
    for feature_idx, sequence_ids in enumerate(seq_ids):
        start_logits = outputs.start_logits[feature_idx]
        end_logits = outputs.end_logits[feature_idx]
        feature_offsets = offsets[feature_idx]
        context_positions = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        for start in context_positions:
            for end in range(start, min(start + MAX_ANSWER_TOKENS, len(sequence_ids) - 1) + 1):
                if sequence_ids[end] != 1:
                    break
                char_start = int(feature_offsets[start][0])
                char_end = int(feature_offsets[end][1])
                if char_end <= char_start:
                    continue
                answer = context[char_start:char_end].strip()
                if not answer or len(answer) > 500:
                    continue
                score = float(start_logits[start].item() + end_logits[end].item())
                if best is None or score > best["score"]:
                    best = {
                        "answer": answer, "score": score,
                        "evidence": context[max(0, char_start - 180):min(len(context), char_end + 220)].strip(),
                        "source": "Saved BERT-CUAD model"
                    }
    if best and best["answer"].lower().strip(" .,:;") not in {"the", "this", "that", "and", "for", "with", "shall", "agreement"}:
        return best
    return None

NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
DURATION_RE = re.compile(
    rf"\b(?:\d+|{NUMBER_WORDS})(?:\s*\(\s*\d+\s*\))?\s+"
    r"(?:calendar\s+|business\s+)?(?:days?|weeks?|months?|years?)\b", re.I
)


@dataclass(frozen=True)
class RawClausePrediction:
    """Best span across all windows before threshold decision."""

    span_confidence: float
    cls_confidence: float
    start_char: int
    end_char: int
    windows_considered: int


class ContractAnalyzer:
    """QA-based clause extraction over contracts."""

    def __init__(
        self,
        model_config_path: Path = MODEL_CONFIG_PATH,
        device: str | None = None,
        quiet: bool = False,
    ):
        self.config = load_model_config(model_config_path)
        self.window_cfg = load_window_config()
        self._load_clause_registry()
        self.device = get_device(device)
        self.quiet = quiet
        self.device = get_device(device)
      
        if not torch.cuda.is_available():
            torch.cuda.is_current_stream_capturing = lambda *a, **k: False

        self._tokenizer, self._model, self.model_info = self._resolve_model()

        threads = self.config.get("performance", {}).get("torch_threads", 8)
        torch.set_num_threads(int(threads))

        self._question_cache: dict[str, list[int]] = {}
        self._tokenized_cache: dict[str, TokenizedContext] = {}

        self._log(f"device: {device_summary(self.device)}")

    # ---------------------------------------------------------- setup

    def _load_clause_registry(self):
        from .config import ConfigError

        try:
            self.clauses = {c.label: c for c in load_enabled_clauses()}
        except ConfigError as exc:
            raise InferenceError(str(exc)) from exc

    def _resolve_model(self):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        final_dir = PROJECT_ROOT / self.config.get(
            "final_model", {}
        ).get("path", "ml/models/final")

        baseline_id = self.config["baseline_model"]["model_id"]
        version = self.config.get("model_version", "unknown")
        attn = "sdpa" if torch.cuda.is_available() else "eager"

        if (final_dir / "config.json").exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(final_dir), use_fast=True
                )
                model = AutoModelForQuestionAnswering.from_pretrained(
                    str(final_dir), attn_implementation=attn
                )
                model.to(self.device)

                info = {
                    "state": "fine_tuned",
                    "source": str(final_dir.relative_to(PROJECT_ROOT)),
                    "model_version": version,
                }
                self._log(f"Loaded FINE-TUNED model from {final_dir}")
                return tokenizer, model, info

            except Exception as exc:
                self._log(
                    f"[warn] final model failed to load ({exc}); "
                    "falling back to baseline"
                )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                baseline_id, use_fast=True
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                baseline_id, attn_implementation=attn
            )
            model.to(self.device)

        except Exception as exc:
            raise InferenceError(
                f"Could not load the baseline QA model '{baseline_id}' ({exc}). "
                "Download it once or place a fine-tuned checkpoint in ml/models/final/."
            ) from exc

        info = {
            "state": "zero_shot_baseline",
            "source": baseline_id,
            "model_version": version,
        }

        self._log(f"Loaded ZERO-SHOT BASELINE model ({baseline_id})")
        return tokenizer, model, info

    def _log(self, message: str):
        if not self.quiet:
            print(f"[ContractAnalyzer] {message}")

    # ---------------------------------------------------------- caching

    def _question_ids(self, label: str) -> list[int]:
        if label not in self._question_cache:
            clause = self.clauses.get(label)

            if clause is None:
                raise KeyError(
                    f"Unknown clause label '{label}'. "
                    f"Enabled: {sorted(self.clauses)}"
                )

            validate_window_fit(
                self.window_cfg,
                self._tokenizer,
                clause.question,
                label,
            )

            self._question_cache[label] = self._tokenizer(
                clause.question,
                add_special_tokens=False,
            )["input_ids"]

        return self._question_cache[label]

    def _tokenized(
        self,
        contract_text: str,
        contract_id: str = "document",
    ) -> TokenizedContext:
        key = f"{contract_id}:{hash(contract_text) & 0xFFFFFFFF}"

        if key not in self._tokenized_cache:
            self._tokenized_cache = {
                key: tokenize_context(
                    self._tokenizer,
                    _Shim(contract_id, contract_text),
                )
            }

        return self._tokenized_cache[key]

    # ---------------------------------------------------------- windows

    def _windows_for_clause(
        self,
        tokenized: TokenizedContext,
        label: str,
    ) -> tuple[list, list]:
        q_ids = self._question_ids(label)
        retrieval = self.config.get("retrieval", {})
        clause = self.clauses[label]

        full_doc = set(retrieval.get("full_document_clauses", []))

        if retrieval.get("enabled") and label not in full_doc:
            regions = select_regions(
                tokenized.context,
                clause,
                top_k=int(retrieval.get("top_k", 15)),
                max_block_chars=int(
                    retrieval.get("max_block_chars", 1200)
                ),
                overlap_chars=int(
                    retrieval.get("overlap_chars", 200)
                ),
                boost_first_blocks=int(
                    retrieval.get("boost_first_blocks", 2)
                ),
            )
            ranges = merge_regions(regions)
        else:
            ranges = [(0, len(tokenized.context))]

        offsets = tokenized.offsets
        windows = []

        for start, end in ranges:
            lo, hi = 0, len(offsets)

            while lo < hi and offsets[lo][1] <= start:
                lo += 1

            while hi > lo and offsets[hi - 1][0] >= end:
                hi -= 1

            if hi <= lo:
                continue

            windows.extend(
                build_windows_for_token_slice(
                    tokenized,
                    lo,
                    hi,
                    q_ids,
                    label,
                    self.window_cfg,
                    self._tokenizer.cls_token_id,
                    self._tokenizer.sep_token_id,
                    first_window_index=len(windows),
                )
            )

        return windows, ranges

    # ---------------------------------------------------------- inference

    def _forward(self, windows: list) -> list[WindowPrediction]:
        batch_size = int(
            self.config.get("performance", {}).get("batch_size", 8)
        )
        max_answer = int(
            self.config["windowing"]["max_answer_tokens"]
        )
        pad_id = self._tokenizer.pad_token_id
        predictions = []

        with torch.inference_mode():
            for i in range(0, len(windows), batch_size):
                batch = windows[i:i + batch_size]
                max_len = max(len(w.input_ids) for w in batch)

                input_ids = torch.tensor(
                    [
                        w.input_ids + [pad_id] * (
                            max_len - len(w.input_ids)
                        )
                        for w in batch
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                attention = torch.tensor(
                    [
                        [1] * len(w.input_ids) + [0] * (
                            max_len - len(w.input_ids)
                        )
                        for w in batch
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                output = self._model(
                    input_ids=input_ids,
                    attention_mask=attention,
                )

                for k, window in enumerate(batch):
                    ctx_start = window.ctx_first_token_index_in_window
                    ctx_end = ctx_start + (
                        window.ctx_token_end - window.ctx_token_start
                    )

                    predictions.append(
                        decode_window(
                            output.start_logits[k],
                            output.end_logits[k],
                            ctx_start,
                            ctx_end,
                            max_answer,
                            window_index=window.window_index,
                        )
                    )

        return predictions

    def _raw_prediction(
        self,
        contract_text: str,
        label: str,
        tokenized: TokenizedContext | None = None,
    ) -> tuple[RawClausePrediction, dict]:

        tokenized = tokenized or self._tokenized(contract_text)
        started = time.perf_counter()

        windows, ranges = self._windows_for_clause(tokenized, label)

        if not windows:
            return (
                RawClausePrediction(0.0, 0.0, -1, -1, 0),
                {"windows": 0, "processing_ms": 0.0, "ranges": []},
            )

        predictions = self._forward(windows)
        best = max(predictions, key=lambda p: p.span_confidence)
        cls_score = max(p.cls_confidence for p in predictions)

        window = next(
            w for w in windows
            if w.window_index == best.window_index
        )

        local_start = best.start_token - window.ctx_first_token_index_in_window
        local_end = best.end_token - window.ctx_first_token_index_in_window

        start_char = window.ctx_offsets[local_start][0]
        end_char = window.ctx_offsets[local_end][1]

        elapsed = round(
            (time.perf_counter() - started) * 1000,
            1,
        )

        return (
            RawClausePrediction(
                round(best.span_confidence, 6),
                round(cls_score, 6),
                start_char,
                end_char,
                len(windows),
            ),
            {
                "windows": len(windows),
                "processing_ms": elapsed,
                "ranges": ranges,
            },
        )

    def _decide(
        self,
        label: str,
        raw: RawClausePrediction,
        context: str,
    ) -> ClauseDecision:

        decoding = self.config["decoding"]

        found = (
            raw.span_confidence
            >= float(decoding.get("min_confidence", 0.0))
            and raw.span_confidence - raw.cls_confidence
            >= float(decoding.get("no_answer_delta", 0.0))
        )

        text = (
            context[raw.start_char:raw.end_char]
            if found and raw.start_char >= 0
            else ""
        )

        return ClauseDecision(
            clause_label=label,
            found=bool(found),
            text=text,
            confidence=raw.span_confidence,
            start_char=raw.start_char if found else -1,
            end_char=raw.end_char if found else -1,
            no_answer_score=raw.cls_confidence,
            window_index=None,
            windows_considered=raw.windows_considered,
        )

    def _result(self, label: str, decision: ClauseDecision, meta: dict):
        result = decision_to_dict(decision)
        result.update(
            question=self.clauses[label].question,
            model_state=self.model_info["state"],
            model_version=self.model_info["model_version"],
            processing_ms=meta["processing_ms"],
        )
        return result

    # ---------------------------------------------------------- public API

    def analyze_clause(
        self,
        contract_text: str,
        clause_label: str,
    ) -> dict:
        tokenized = self._tokenized(contract_text)
        raw, meta = self._raw_prediction(
            contract_text,
            clause_label,
            tokenized,
        )
        return self._result(
            clause_label,
            self._decide(clause_label, raw, tokenized.context),
            meta,
        )
    
    def answer_question(
        self,
        contract_text: str,
        question: str,
    ) -> dict:
        """
        Answer a user question using date/duration rules first,
        then fall back to the saved BERT QA model.
        """
        if not contract_text or not contract_text.strip():
            raise ValueError("Contract text cannot be empty.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        # 1. Try explicit date/duration extraction on the full text.
        result = targeted_duration_answer(question, contract_text)

        if result is None:
            result = targeted_date_answer(question, contract_text)

        # 2. If rules find nothing, use the already-loaded BERT model.
        if result is None:
            result = bert_answer(
                question,
                contract_text,
                self._tokenizer,
                self._model,
                self.device,
            )

        # 3. Return a consistent Q&A response.
        if result is None:
            return {
                "question": question,
                "answer": "",
                "evidence": "",
                "source": "No reliable answer found",
                "found": False,
                "model_state": self.model_info["state"],
                "model_version": self.model_info["model_version"],
            }

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "evidence": result.get("evidence", ""),
            "source": result.get("source", "Saved BERT-CUAD model"),
            "found": bool(result.get("answer", "").strip()),
            "model_state": self.model_info["state"],
            "model_version": self.model_info["model_version"],
        }

    def iter_raw(
        self,
        contract_text: str,
        enabled_clauses: list[str] | None = None,
    ):
        labels = enabled_clauses or list(self.clauses)
        tokenized = self._tokenized(contract_text)

        for label in labels:
            yield (
                label,
                *self._raw_prediction(
                    contract_text,
                    label,
                    tokenized,
                ),
            )

    def analyze_contract_iter(
        self,
        contract_text: str,
        enabled_clauses: list[str] | None = None,
    ):
        labels = enabled_clauses or list(self.clauses)
        tokenized = self._tokenized(contract_text)
        started = time.perf_counter()

        for label in labels:
            raw, meta = self._raw_prediction(
                contract_text,
                label,
                tokenized,
            )

            yield self._result(
                label,
                self._decide(label, raw, tokenized.context),
                meta,
            )

        yield {
            "_summary": {
                "total_processing_ms": round(
                    (time.perf_counter() - started) * 1000,
                    1,
                ),
                "clauses": len(labels),
                "model_state": self.model_info["state"],
                "model_version": self.model_info["model_version"],
            }
        }

    def analyze_contract(
        self,
        contract_text: str,
        enabled_clauses: list[str] | None = None,
    ) -> dict:
        results = []
        summary = {}

        for item in self.analyze_contract_iter(
            contract_text,
            enabled_clauses,
        ):
            if "_summary" in item:
                summary = item["_summary"]
            else:
                results.append(item)

        return {
            "clauses": results,
            "model_state": summary.get(
                "model_state",
                self.model_info["state"],
            ),
            "model_version": summary.get(
                "model_version",
                self.model_info["model_version"],
            ),
            "total_processing_ms": summary.get("total_processing_ms"),
            "disclaimer": (
                "AI-assisted analysis. Results should be reviewed "
                "by a qualified professional."
            ),
        }
    def clear_cache(self):
        """Clear the internal tokenization and question caches."""
        self._tokenized_cache.clear()
        self._question_cache.clear()

    def question(self, clause_label: str) -> str:
        """Return the question associated with a clause label."""
        clause = self.clauses.get(clause_label)
        if clause is None:
            raise KeyError(
                f"Unknown clause label '{clause_label}'. "
                f"Enabled: {sorted(self.clauses)}"
            )
        return clause.question

class _Shim:
    """Minimal object satisfying tokenize_context's record interface."""

    def __init__(self, contract_id: str, context: str):
        self.contract_id = contract_id
        self.context = context

      

