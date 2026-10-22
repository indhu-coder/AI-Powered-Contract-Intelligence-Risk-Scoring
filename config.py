"""Central configuration for the ContractIQ ML pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_CONFIG_DIR = PROJECT_ROOT / "ml" / "configs"
CLAUSE_REGISTRY_PATH = ML_CONFIG_DIR / "clauses.json"


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CUADV1_PATH = RAW_DIR / "CUADv1.json"
TRAIN_SEPARATE_PATH = RAW_DIR / "train_separate_questions.json"
TEST_PATH = RAW_DIR / "test.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TOKENIZER_NAME = "distilbert-base-uncased"
DEFAULT_SEED = 42
DEFAULT_VAL_FRACTION = 0.10


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class ClauseSpec:
    label: str
    cuad_category: str
    question: str
    enabled: bool


@dataclass(frozen=True)
class WindowConfig:
    max_seq_len: int
    doc_stride: int
    max_answer_tokens: int
    no_answer_threshold: float
    min_confidence: float

    def __post_init__(self):
        if not 32 <= self.max_seq_len <= 2048:
            raise ConfigError(
                f"max_seq_len out of range: {self.max_seq_len}"
            )

        if not 0 < self.doc_stride < self.max_seq_len - 16:
            raise ConfigError(
                f"doc_stride must be in (0, max_seq_len-16); "
                f"got {self.doc_stride}"
            )

        if not 0 <= self.no_answer_threshold <= 1:
            raise ConfigError(
                f"no_answer_threshold must be in [0, 1]"
            )

    def check_tokenizer_compat(self, model_max_length: int):
        if not model_max_length or model_max_length <= 0:
            return

        if self.max_seq_len > model_max_length:
            raise ConfigError(
                f"max_seq_len={self.max_seq_len} exceeds "
                f"tokenizer model_max_length={model_max_length}; "
                "lower max_seq_len in ml/configs/clauses.json"
            )


@dataclass(frozen=True)
class EnabledClause(ClauseSpec):
    """Clause enabled for Version 1."""


def load_clause_registry(
    path: Path = CLAUSE_REGISTRY_PATH,
) -> tuple[list[ClauseSpec], dict]:
    """Load and validate the central clause registry."""

    if not path.exists():
        raise ConfigError(
            f"Clause registry not found at {path}. "
            "Regenerate it with: "
            "python scripts/generate_clause_config.py"
        )

    registry = json.loads(path.read_text(encoding="utf-8"))

    clauses = [
        ClauseSpec(
            label=c.get("label", ""),
            cuad_category=c["cuad_category"],
            question=c["question"],
            enabled=bool(c.get("enabled", False)),
        )
        for c in registry.get("clauses", [])
    ]

    if not clauses:
        raise ConfigError("Clause registry contains no clauses.")

    enabled = [c for c in clauses if c.enabled]
    labels = [c.label for c in enabled]

    if len(labels) != len(set(labels)):
        raise ConfigError("Duplicate labels among enabled clauses.")

    for clause in enabled:
        if not clause.label:
            raise ConfigError(
                f"Enabled clause '{clause.cuad_category}' has no label."
            )

    return clauses, registry


def load_enabled_clauses(
    path: Path = CLAUSE_REGISTRY_PATH,
) -> list[EnabledClause]:
    """Return enabled clauses in registry order."""

    clauses, _ = load_clause_registry(path)

    return [
        EnabledClause(
            c.label,
            c.cuad_category,
            c.question,
            True,
        )
        for c in clauses
        if c.enabled
    ]


def load_window_config(
    path: Path = CLAUSE_REGISTRY_PATH,
) -> WindowConfig:
    """Load inference/window settings from the central registry."""

    _, registry = load_clause_registry(path)
    config = registry.get("inference", {})

    try:
        return WindowConfig(
            max_seq_len=int(config["max_seq_len"]),
            doc_stride=int(config["doc_stride"]),
            max_answer_tokens=int(config["max_answer_tokens"]),
            no_answer_threshold=float(
                config["no_answer_threshold"]
            ),
            min_confidence=float(config["min_confidence"]),
        )
    except KeyError as exc:
        raise ConfigError(
            f"Missing inference config key in {path}: {exc}"
        ) from exc
