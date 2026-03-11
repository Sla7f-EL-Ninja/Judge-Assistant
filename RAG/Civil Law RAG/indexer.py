"""
indexer.py

Civil Law indexing pipeline.

Purpose:
---------
Responsible for one-time indexing of the Egyptian Civil Law
into the vector database.

Pipeline:
---------
1. Load raw text file
2. Split using hierarchical splitter
3. Batch embed documents
4. Persist into Chroma database

Important:
----------
This module should NOT run automatically during application runtime.
Indexing is a controlled operation executed manually when:

- The law file changes
- Embedding model changes
- Database needs to be rebuilt

Why this exists:
----------------
Indexing is computationally expensive and must be isolated from
query-time logic.

Design Principle:
-----------------
Clear separation between:
- Offline indexing stage
- Online retrieval & reasoning stage
"""
import os
from langchain_community.document_loaders import TextLoader
from config import DOCS_PATH, DB_DIR, BATCH_SIZE
from splitter import split_egyptian_civil_law
from vectorstore import load_vectorstore


def index_civil_law():
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"{DOCS_PATH} not found")

    db = load_vectorstore()

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Vectorstore already exists. Skipping indexing.")
        return db  # return existing db

    print("Indexing Egyptian Civil Law...")

    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    document = loader.load()

    raw_text = document[0].page_content
    docs = split_egyptian_civil_law(raw_text)

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"Indexed {i + len(batch)} / {len(docs)}")

    db.persist()
    print("Indexing completed.")
    return db
