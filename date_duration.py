import json
import re
import pandas as pd

# ============================================================
# LOAD CUAD
# ============================================================

CUAD_PATH = r"D:\\ContractIntelligence\\cuad-main\\data\\CUADv1.json"

with open(CUAD_PATH, "r", encoding="utf-8") as f:
    cuad = json.load(f)

# ============================================================
# KEYWORDS FOR DATE / DURATION QUESTIONS
# ============================================================

date_duration_keywords = [
    "date",
    "dates",
    "effective",
    "expiration",
    "expire",
    "expiry",
    "termination date",
    "notice period",
    "notice",
    "renewal",
    "renew",
    "term",
    "duration",
    "days",
    "months",
    "years",
    "deadline",
    "commencement"
]

# ============================================================
# EXTRACT FROM CUAD
# ============================================================

examples = []

for document in cuad["data"]:

    for paragraph in document["paragraphs"]:

        context = paragraph["context"]

        for qa in paragraph["qas"]:

            question = qa["question"]

            question_lower = question.lower()

            # ------------------------------------------------
            # Keep only date/duration-related questions
            # ------------------------------------------------

            if not any(
                keyword in question_lower
                for keyword in date_duration_keywords
            ):
                continue

            # ------------------------------------------------
            # Skip impossible questions
            # ------------------------------------------------

            if qa.get("is_impossible", False):
                continue

            answers = qa.get("answers", [])

            if not answers:
                continue

            # ------------------------------------------------
            # Keep answer + answer_start
            # ------------------------------------------------

            answer = answers[0]

            answer_text = answer["text"]
            answer_start = answer["answer_start"]

            examples.append({

                "context": context,

                "question": question,

                "answer": answer_text,

                "answer_start": answer_start,

                "answer_end":
                    answer_start + len(answer_text)

            })

# ============================================================
# CREATE DATAFRAME
# ============================================================

df_date_duration = pd.DataFrame(examples)

print(
    "Date/Duration examples:",
    len(df_date_duration)
)

print(
    df_date_duration[
        [
            "question",
            "answer",
            "answer_start",
            "answer_end"
        ]
    ].head(5)
)

OUTPUT_FILE = r"D:\\ContractIntelligence\\cuad_date_duration.csv"

df_date_duration.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:", OUTPUT_FILE)