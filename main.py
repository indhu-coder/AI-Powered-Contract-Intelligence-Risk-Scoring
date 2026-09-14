import json
import pandas as pd
from dateutil import parser
import joblib
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load JSON
with open(r"D:\\ContractIntelligence\\cuad-main\\data\\CUADv1.json", "r", encoding="utf-8") as f:
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
print(df.head())
print("\nShape:", df.shape)
print(df.isnull().sum())
print("\nColumns:")
print(df.columns.tolist())
print(df.info())

field_counts = df['field'].value_counts()
print("\nField Counts:")
print(field_counts)

title_counts = df['contract_title'].value_counts()
print("\nContract Title Counts:")   
print(title_counts)

# Date analyis
# ---------------- DATE ANALYSIS ----------------

# ---------------- DATE ANALYSIS ----------------

import pandas as pd
from dateutil import parser
import re

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


print("\nDATE ANALYSIS")
print(
    df[df["field"].isin(date_fields)]
    [["contract_title", "field", "extracted_date"]]
    .head(20)
)

print("\nDURATION ANALYSIS")
print(
    df[df["field"].isin(duration_fields)]
    [["field", "duration"]]
    .head(20)
    .to_string(index=False)
)

print("\nDuration Summary")
print(df.groupby("field")["duration"].count())
print(
    df[df["duration"].notna()]
    [["field", "duration"]]
    .value_counts()
)
#Tokenization and Preprocessing


# def preprocess_text(text):
#     if not text:
#         return ""
#     # Lowercase
#     text = text.lower()
#     # Remove punctuation and special characters
#     text = re.sub(r'[^\w\s]', '', text)
#     # Tokenize
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#     return tokens

# df["tokenized_question"] = df["question"].apply(preprocess_text)
# df["tokenized_context"] = df["context_text"].apply(preprocess_text)

# df["tokenized_answer"] = df["answer_text"].apply(preprocess_text)


# # -------------------------------
# # LOAD SPACY MODEL
# # -------------------------------

# nlp = spacy.load("en_core_web_md")

# embeddings = []

# for idx, row in df.iterrows():

#     # Use ORIGINAL text, not tokenized text
#     question_text = row["question"] if pd.notna(row["question"]) else ""
#     context_text = row["context_text"] if pd.notna(row["context_text"]) else ""
#     answer_text = row["answer_text"] if pd.notna(row["answer_text"]) else ""

#     # Generate documents
#     question_doc = nlp(question_text)
#     context_doc = nlp(context_text)
#     answer_doc = nlp(answer_text)

#     # Store embeddings
#     embeddings.append({
#         "qa_id": row["qa_id"],
#         "field": row["field"],
#         "question_embedding": question_doc.vector,
#         "context_embedding": context_doc.vector,
#         "answer_embedding": answer_doc.vector
#     })

# print("\nEmbeddings generated successfully.")
# print("Total embeddings generated:", len(embeddings))

# if embeddings:
#     print(
#         "Question embedding shape:",
#         embeddings[0]["question_embedding"].shape
#     )



