import json
import pandas as pd
from dateutil import parser
import joblib
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize


from pathlib import Path

# Load JSON
BASE_DIR = Path(__file__).resolve().parent
json_path = BASE_DIR / "cuad-main" / "data" / "CUADv1.json"
if not json_path.exists():
    fallback = Path(r"D:\ContractIntelligence\cuad-main\data\CUADv1.json")
    if fallback.exists():
        json_path = fallback
    else:
        raise FileNotFoundError(f"CUADv1.json not found at {json_path}")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for contract in data["data"]:
    contract_title = contract.get("title", "")
    contract_title = contract_title.replace("-", "_")
    contract_title = contract_title.split("_")[-1]  # Use last part of title as identifier
    for paragraph in contract.get("paragraphs", []):

        # The JSON sample does not show a context field
        context = paragraph.get("context", "")

        for qa in paragraph.get("qas", []):
            qa_id = qa.get("id", "")
            details = qa.get("details", {})
            # Extract field name from ID
            field = qa_id.split("__")[-1]
            
            answers = qa.get("answers", [])

            # Multiple answers → multiple rows
            if answers:
                for answer in answers:
                    rows.append({
                        "contract_title": contract_title,
                        "qa_id": qa_id,
                        "details": details,
                        "field": field,
                        "question": qa.get("question", ""),
                        "context_text": context,
                        "answer_text": answer.get("text", ""),
                        "answer_start": answer.get("answer_start", None),
                        "is_impossible": qa.get("is_impossible", False)
                    })

            # No answer / impossible question
            else:
                rows.append({
                    "contract_title": contract_title,
                    "qa_id": qa_id,
                    "details": details,
                    "field": field,
                    "question": qa.get("question", ""),
                    "context_text": context,
                    "answer_text": "",
                    "answer_start": None,
                    "is_impossible": qa.get("is_impossible", True)
                })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Basic inspection
# print(df.head())
# print("\nShape:", df.shape)
# print(df.isnull().sum())
# print("\nColumns:")
# print(df.columns.tolist())
# print(df.info())

# field_counts = df['field'].value_counts()
# print("\nField Counts:")
# print(field_counts)

# title_counts = df['contract_title'].value_counts()
# print("\nContract Title Counts:")   
# print(title_counts)


# ---------------- DATE ANALYSIS ----------------


date_fields = [
    "Agreement Date",
    "Effective Date",
    "Expiration Date"
]

duration_fields = [
    "Renewal Term",
    "Notice Period To Terminate Renewal"
]

def extract_date(text):
    if pd.isna(text) or str(text).strip() == "":
        return pd.NaT

    try:
        return pd.Timestamp(parser.parse(str(text), fuzzy=True))
    except:
        return pd.NaT
    
def extract_duration(text):
    if pd.isna(text) or not str(text).strip():
        return None

    text = str(text)

    # Prefer number inside parentheses: one hundred eighty (180) days
    match = re.search(
        r'\((\d+)\)\s*(day|days|week|weeks|month|months|year|years)\b',
        text,
        re.IGNORECASE
    )

    if match:
        return f"{match.group(1)} {match.group(2)}"

    # Normal numeric form: 30 days
    match = re.search(
        r'\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b',
        text,
        re.IGNORECASE
    )

    return match.group(0) if match else None


# Create separate results
date_result = df["answer_text"].where(
    df["field"].isin(date_fields)
).apply(extract_date)

duration_result = df["answer_text"].where(
    df["field"].isin(duration_fields)
).apply(extract_duration)

df["duration"] = duration_result


# Assign after extraction
df["extracted_date"] = date_result
df["duration"] = duration_result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CUAD DATASET INSPECTION")
    print("=" * 60)
    print("DataFrame Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    
    print("\nTop 10 Fields:")
    print(df["field"].value_counts().head(10))

    print("\n" + "=" * 60)
    print("DATE ANALYSIS")
    print("=" * 60)
    date_df = df[df["field"].isin(date_fields)][["contract_title", "field", "extracted_date"]].dropna(subset=["extracted_date"])
    print(date_df.head(15).to_string(index=False))

    print("\n" + "=" * 60)
    print("DURATION ANALYSIS")
    print("=" * 60)
    duration_df = df[df["field"].isin(duration_fields)][["field", "duration"]].dropna(subset=["duration"])
    print(duration_df.head(15).to_string(index=False))

    print("\nDuration Counts by Field:")
    print(df.groupby("field")["duration"].count())

    # -------------------------------
    # DEMO SPACY EMBEDDINGS (SAMPLE)
    # -------------------------------
    print("\n" + "=" * 60)
    print("SPACY EMBEDDINGS (SAMPLE)")
    print("=" * 60)
    try:
        nlp = spacy.load("en_core_web_sm")
        print("SpaCy model 'en_core_web_sm' loaded successfully.")
    except Exception as e:
        print(f"Could not load spaCy model: {e}")
        nlp = None

    if nlp is not None:
        sample_df = df.head(10)
        embeddings = []
        for idx, row in sample_df.iterrows():
            q_text = row["question"] if pd.notna(row["question"]) else ""
            c_text = row["context_text"][:200] if pd.notna(row["context_text"]) else ""
            a_text = row["answer_text"] if pd.notna(row["answer_text"]) else ""

            embeddings.append({
                "qa_id": row["qa_id"],
                "field": row["field"],
                "question_embedding": nlp(q_text).vector,
                "context_embedding": nlp(c_text).vector,
                "answer_embedding": nlp(a_text).vector
            })

        print(f"Sample embeddings generated: {len(embeddings)}")
        print("Vector dimension:", embeddings[0]["question_embedding"].shape)
        print("=" * 60)
        print("Pipeline main.py finished successfully.")



