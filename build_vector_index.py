"""ContractIQ - Build Persistent FAISS Vector Index from CUAD Contracts.

Loads contracts from CUADv1.json, chunks text segments, encodes embeddings
using SentenceTransformer, and writes a persistent FAISS index to disk.

Usage:
    python build_vector_index.py --limit 100
    python build_vector_index.py   # Full index
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from vector_store import VectorStore

BASE_DIR = Path(__file__).resolve().parent
CUAD_PATH = BASE_DIR / "cuad-main" / "data" / "CUADv1.json"
INDEX_DIR = BASE_DIR / "data" / "vector_store"


def prepare_cuad_chunks(limit_contracts: int | None = None, chunk_size: int = 800) -> List[Dict]:
    """Break contracts into semantic passages for granular vector retrieval."""
    if not CUAD_PATH.exists():
        raise FileNotFoundError(f"CUAD dataset not found at {CUAD_PATH}")

    with open(CUAD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    contracts = data.get("data", [])
    if limit_contracts:
        contracts = contracts[:limit_contracts]

    docs: List[Dict] = []
    chunk_counter = 0

    for item in contracts:
        raw_title = item.get("title", "")
        clean_title = raw_title.replace("-", "_").split("_")[-1]

        for p_idx, p in enumerate(item.get("paragraphs", [])):
            ctx = p.get("context", "").strip()
            if not ctx:
                continue

            # Slide over context in chunk_size windows with 200 char overlap
            step = max(300, chunk_size - 200)
            for i in range(0, len(ctx), step):
                passage = ctx[i : i + chunk_size].strip()
                if len(passage) < 80:
                    continue

                chunk_counter += 1
                docs.append({
                    "id": f"chunk_{chunk_counter}",
                    "contract_title": clean_title,
                    "raw_title": raw_title,
                    "paragraph_index": p_idx,
                    "char_start": i,
                    "char_end": i + len(passage),
                    "text": passage,
                })

    return docs


def build_index(limit: int | None = None) -> None:
    print("=" * 60)
    print("CONTRACTIQ: POPULATING FAISS VECTOR DATABASE")
    print(f"Source: {CUAD_PATH}")
    print(f"Destination: {INDEX_DIR}")
    print(f"Limit Contracts: {limit or 'ALL'}")
    print("=" * 60)

    t0 = time.time()
    print("Preparing contract text chunks...")
    chunks = prepare_cuad_chunks(limit_contracts=limit)
    print(f"Extracted {len(chunks)} searchable passages.")

    print("Initializing FAISS Vector Store...")
    store = VectorStore()

    print(f"Generating dense embeddings using {store.model_name}...")
    n_added = store.add_documents(chunks, batch_size=64, show_progress=True)

    print(f"Saving vector database to {INDEX_DIR}...")
    store.save(INDEX_DIR)

    elapsed = round(time.time() - t0, 2)
    print("=" * 60)
    print(f"Vector Database built successfully in {elapsed}s!")
    print(f"Total Vectors Indexed: {store.count()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Vector Index")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of contracts for indexing (default: 50 for fast build)")
    args = parser.parse_args()

    build_index(limit=args.limit)


if __name__ == "__main__":
    main()
