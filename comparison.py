import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import re
import string
from collections import Counter

MODEL_DIR = r"/kaggle/input/models/indhumathythangavelu/bert-cuad-qa/transformers/default/4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print("Model loaded:", MODEL_DIR)
print("Device:", device)

results_df = pd.read_csv("/kaggle/input/datasets/indhumathythangavelu/bert-100-validation-prediction/bert_validation_predictions.csv")

print(results_df.columns.tolist())
print(results_df.head())
      
def normalize_answer(text):
    text = str(text).lower()
    text = "".join(
        ch for ch in text if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(reference, prediction):
    return float(
        normalize_answer(reference) ==
        normalize_answer(prediction)
    )


def token_f1(reference, prediction):
    ref_tokens = normalize_answer(reference).split()
    pred_tokens = normalize_answer(prediction).split()

    if not ref_tokens or not pred_tokens:
        return float(ref_tokens == pred_tokens)

    common = Counter(ref_tokens) & Counter(pred_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


df = results_df.copy()

required = [
    "reference", "candidate", "null_minus_candidate"
]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["reference"] = df["reference"].fillna("").astype(str)
df["candidate"] = df["candidate"].fillna("").astype(str)
df["null_minus_candidate"] = pd.to_numeric(
    df["null_minus_candidate"], errors="raise"
)

rows = []

for threshold in np.arange(-5.0, 5.01, 0.25):
    predictions = [
        "" if score > threshold else answer
        for score, answer in zip(
            df["null_minus_candidate"], df["candidate"]
        )
    ]

    em = np.mean([
        exact_match(r, p)
        for r, p in zip(df["reference"], predictions)
    ])

    f1 = np.mean([
        token_f1(r, p)
        for r, p in zip(df["reference"], predictions)
    ])

    rows.append({
        "threshold": round(float(threshold), 2),
        "EM": round(float(em), 4),
        "F1": round(float(f1), 4)
    })

threshold_results = pd.DataFrame(rows)

print(
    threshold_results.sort_values("F1", ascending=False).head(10)
)

best_row = threshold_results.loc[
    threshold_results["F1"].idxmax()
]

BEST_THRESHOLD = float(best_row["threshold"])

print("\nBest validation threshold:", BEST_THRESHOLD)
print("Validation EM:", best_row["EM"])
print("Validation F1:", best_row["F1"])

threshold_results.to_csv(
    "validation_threshold_comparison.csv", index=False
)
