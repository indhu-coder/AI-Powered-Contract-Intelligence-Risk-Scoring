import pandas as pd
import numpy as np
from nerpipeline import df
from sentence_transformers import SentenceTransformer

####   Embeddings

print("\n" + "=" * 60)
print("TRANSFORMER EMBEDDINGS (SentenceTransformer)")
print("=" * 60)
print("Loading model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def entities_to_text(entities):
    if not entities or not isinstance(entities, list):
        return ""
    return " ".join(
        entity.get("text", "") if isinstance(entity, dict) else str(entity)
        for entity in entities
    )

def contract_title_to_text(title):
    if not title or pd.isna(title):
        return ""
    return str(title)

df["entities_text"] = df["legal_entities"].apply(entities_to_text)
df["contract_title_text"] = df["contract_title"].apply(contract_title_to_text)

# Encode unique entity texts for fast vectorization
unique_entities = [t for t in df["entities_text"].unique() if t]
print(f"Encoding {len(unique_entities)} unique entity summaries...")
entity_vectors = model.encode(unique_entities, show_progress_bar=True, batch_size=64)
entity_map = dict(zip(unique_entities, list(entity_vectors)))
df["entities_embedding"] = df["entities_text"].map(lambda x: entity_map.get(x, np.zeros(384)))

# Encode unique contract titles
unique_titles = [t for t in df["contract_title_text"].unique() if t]
print(f"Encoding {len(unique_titles)} unique contract titles...")
title_vectors = model.encode(unique_titles, show_progress_bar=True, batch_size=64)
title_map = dict(zip(unique_titles, list(title_vectors)))
df["contract_title_embedding"] = df["contract_title_text"].map(lambda x: title_map.get(x, np.zeros(384)))

print("Embeddings generated successfully.")
print("Entity embeddings shape:", np.array(df["entities_embedding"].tolist()).shape)
print("Contract title embeddings shape:", np.array(df["contract_title_embedding"].tolist()).shape)
