
import json
import re

import pandas as pd
import spacy
from dateutil import parser


# -------------------------------
# CONFIGURATION
# -------------------------------

DATA_PATH = r"D:\ContractIntelligence\cuad-main\data\CUADv1.json"
SPACY_MODEL = "en_core_web_md"

DATE_FIELDS = [
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
]

DURATION_FIELDS = [
    "Renewal Term",
    "Notice Period To Terminate Renewal",
]


# -------------------------------
# LOAD AND TRANSFORM CUAD DATA
# -------------------------------

def load_cuad_data(file_path):
    """Load CUAD JSON and convert question-answer pairs into rows."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    rows = []

    for contract in data["data"]:
        title = contract.get("title", "")
        title = title.replace("-", "_").split("_")[-1]

        for paragraph in contract.get("paragraphs", []):
            context = paragraph.get("context", "")

            for qa in paragraph.get("qas", []):
                qa_id = qa.get("id", "")
                answers = qa.get("answers", [])
                common = {
                    "contract_title": title,
                    "qa_id": qa_id,
                    "details": qa.get("details", {}),
                    "field": qa_id.split("__")[-1],
                    "question": qa.get("question", ""),
                    "context_text": context,
                    "is_impossible": qa.get(
                        "is_impossible", not bool(answers)
                    ),
                }

                if answers:
                    for answer in answers:
                        rows.append({
                            **common,
                            "answer_text": answer.get("text", ""),
                            "answer_start": answer.get("answer_start"),
                        })
                else:
                    rows.append({
                        **common,
                        "answer_text": "",
                        "answer_start": None,
                    })

    return pd.DataFrame(rows)


# -------------------------------
# DATE AND DURATION EXTRACTION
# -------------------------------

def extract_date(value):
    """Parse a date from text; return NaT if parsing fails."""
    if pd.isna(value) or not str(value).strip():
        return pd.NaT

    try:
        return pd.Timestamp(parser.parse(str(value), fuzzy=True))
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def extract_duration(value):
    """Extract numeric day, week, month, or year durations."""
    if pd.isna(value) or not str(value).strip():
        return None

    value = str(value)

    patterns = [
        r"\((\d+)\)\s*(day|days|week|weeks|month|months|year|years)\b",
        r"\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            if pattern == patterns[0]:
                return f"{match.group(1)} {match.group(2)}"
            return match.group(0)

    return None


def preprocess_dates_and_durations(df):
    """Add extracted date and duration columns to the DataFrame."""
    date_values = df["answer_text"].where(
        df["field"].isin(DATE_FIELDS)
    )
    duration_values = df["answer_text"].where(
        df["field"].isin(DURATION_FIELDS)
    )

    df["extracted_date"] = date_values.apply(extract_date)
    df["duration"] = duration_values.apply(extract_duration)

    return df


# -------------------------------
# GENERATE TEXT EMBEDDINGS
# -------------------------------

def generate_embeddings(df, model_name=SPACY_MODEL):
    """Generate spaCy vectors for each question, context, and answer."""
    nlp = spacy.load(model_name)
    print("\nSpaCy model loaded successfully.")

    embeddings = []

    for _, row in df.iterrows():
        question = row["question"] if pd.notna(row["question"]) else ""
        context = row["context_text"] if pd.notna(row["context_text"]) else ""
        answer = row["answer_text"] if pd.notna(row["answer_text"]) else ""

        embeddings.append({
            "qa_id": row["qa_id"],
            "field": row["field"],
            "question_embedding": nlp(question).vector,
            "context_embedding": nlp(context).vector,
            "answer_embedding": nlp(answer).vector,
        })

    print("\nEmbeddings generated successfully.")
    print("Total embeddings generated:", len(embeddings))

    if embeddings:
        print(
            "Question embedding shape:",
            embeddings[0]["question_embedding"].shape,
        )

    return embeddings


# -------------------------------
# MAIN WORKFLOW
# -------------------------------

def main():
    df = load_cuad_data(DATA_PATH)
    df = preprocess_dates_and_durations(df)

    print("Dataset shape:", df.shape)
    print("Date and duration extraction completed.")

    embeddings = generate_embeddings(df)

    return df, embeddings


if __name__ == "__main__":
    df, embeddings = main()
