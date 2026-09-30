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

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])

print("SpaCy NER model loaded successfully (optimized: NER only).")

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

import os
import json
from pathlib import Path

cache_file = Path(__file__).resolve().parent / "ner_entities_cache.json"
entity_map = {}

if cache_file.exists():
    print(f"Loading cached legal entities from {cache_file.name}...")
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            entity_map = json.load(f)
        print(f"Loaded {len(entity_map)} cached context entities.")
    except Exception as e:
        print(f"Could not load cache: {e}, recomputing...")
        entity_map = {}

if not entity_map:
    # Process unique contexts
    limit_str = os.environ.get("NER_LIMIT")
    texts_to_process = unique_texts[:int(limit_str)] if limit_str else unique_texts

    for i, doc in enumerate(
        nlp.pipe(texts_to_process, batch_size=32)
    ):
        entities = [
            {
                "text": ent.text,
                "label": ent.label_
            }
            for ent in doc.ents
            if ent.label_ in legal_labels
        ]

        entity_map[texts_to_process[i]] = entities

        if (i + 1) % 50 == 0 or (i + 1) == len(texts_to_process):
            print(
                f"Processed {i + 1}/{len(texts_to_process)} contexts"
            )

    if not limit_str and len(entity_map) == len(unique_texts):
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entity_map, f)
            print(f"Saved entity cache to {cache_file.name}")
        except Exception as e:
            print(f"Could not save cache: {e}")

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