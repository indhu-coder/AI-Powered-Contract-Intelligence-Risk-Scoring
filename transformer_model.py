import pandas as pd
import numpy as np
from nerpipeline import *
from main import df
from sentence_transformers import SentenceTransformer

####   Embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

def entities_to_text(entities):
    if not entities:
        return ""

    return " ".join(
        entity["text"]
        for entity in entities
    )

def contract_title_to_text(title):
    if not title:
        return ""

    return title
df["entities_text"] = df["legal_entities"].apply(
    entities_to_text
)

df["contract_title_text"] = df["contract_title"].apply(
    contract_title_to_text
)

df["entities_embedding"] = list(
    model.encode(
        df["entities_text"].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
)

df["contract_title_embedding"] = list(
    model.encode(
        df["contract_title_text"].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
)

print("Embeddings generated successfully.")
print("Entity embeddings shape:", np.array(df["entities_embedding"].tolist()).shape)
print("Contract title embeddings shape:", np.array(df["contract_title_embedding"].tolist()).shape)
