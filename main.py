import json
import pandas as pd

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
# print("\nColumns:")
# print(df.columns.tolist())
# print(df.info())

# field_counts = df['field'].value_counts()
# print("\nField Counts:")
# print(field_counts)

# title_counts = df['contract_title'].value_counts()
# print("\nContract Title Counts:")   
# print(title_counts)