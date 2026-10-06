"""Chroma-backed vector store, embedded locally with sentence-transformers
(no external embedding API required — only generation goes through Claude)."""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        _collection = get_client().get_or_create_collection(
            name="papers",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection() -> None:
    """Used by tests / a clean re-index."""
    global _collection
    get_client().delete_collection("papers")
    _collection = None
