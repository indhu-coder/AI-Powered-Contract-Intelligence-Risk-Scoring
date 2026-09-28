from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch
from nerpipeline import *
##Loading the model and tokenizer for LegalBERT
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def extract_legal_entities(text):
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
        if ent.label_ in legal_labels
    ]



legal_labels = {
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period",
    "Governing Law",
    "Jurisdiction",
    "Termination",
    "Confidentiality",
    "Payment Terms",
    "Insurance",
    "Indemnification",
    "Assignment"
}


df['legal_entities'] = df['context_text'].apply(extract_legal_entities)

print(df['legal_entities'].head())

output_embeddings = []

out = model(**tokenizer(df['legal_entities'].apply(lambda x: ' '.join([entity['text'] for entity in x])).tolist(), padding=True, truncation=True, return_tensors='pt'))

output_embeddings.append(out.last_hidden_state)

print("LegalBERT embeddings generated successfully.")

print("LegalBERT embeddings shape:", np.array(output_embeddings).shape)

print("Contract title embeddings shape:", np.array(df["contract_title_embedding"].tolist()).shape)

