import pandas as pd
from main import df
import spacy
from datetime import datetime

# -------------------------------
# START TIME
# -------------------------------

start_time = datetime.now()

print("=" * 60)
print("NER PROCESS STARTED")
print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 60)
# -------------------------------
# SPACY NER
# -------------------------------

nlp = spacy.load("en_core_web_sm")

print("SpaCy NER model loaded successfully.")

legal_labels = {
    "ORG",
    "PERSON",
    "GPE",
    "LOC",
    "DATE",
    "MONEY",
    "LAW"
}

# --------------------------------
# Process UNIQUE contexts only
# --------------------------------

unique_texts = (
    df["context_text"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

print("Total dataframe rows:", len(df))
print("Unique contexts:", len(unique_texts))

entity_map = {}

# Process unique contexts
for i, doc in enumerate(
    nlp.pipe(unique_texts, batch_size=32)
):

    entities = [
        {
            "text": ent.text,
            "label": ent.label_
        }
        for ent in doc.ents
        if ent.label_ in legal_labels
    ]

    entity_map[unique_texts[i]] = entities

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(unique_texts)} contexts"
        )

# --------------------------------
# Map entities back to dataframe
# --------------------------------

df["legal_entities"] = (
    df["context_text"]
    .map(entity_map)
)

print("NER completed.")

# -------------------------------
# END TIME
# -------------------------------

end_time = datetime.now()

elapsed_time = end_time - start_time

print("=" * 60)
print("NER COMPLETED")
print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Total time:", elapsed_time)
print("=" * 60)

# Check results
print(
    df[
        ["contract_title", "legal_entities"]
    ].head(1).to_string(index=False)
)