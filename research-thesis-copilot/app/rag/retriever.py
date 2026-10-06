"""Thin retrieval API used by every agent node — keeps RAG logic in one place."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.rag.vectorstore import get_collection


@dataclass
class RetrievedChunk:
    text: str
    arxiv_id: str
    title: str
    page: int
    pdf_url: str
    distance: float


def retrieve(query: str, k: int = 6, arxiv_id: Optional[str] = None) -> List[RetrievedChunk]:
    collection = get_collection()
    where = {"arxiv_id": arxiv_id} if arxiv_id else None
    result = collection.query(query_texts=[query], n_results=k, where=where)

    chunks: List[RetrievedChunk] = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            RetrievedChunk(
                text=doc,
                arxiv_id=meta.get("arxiv_id", ""),
                title=meta.get("title", ""),
                page=meta.get("page", 0),
                pdf_url=meta.get("pdf_url", ""),
                distance=dist,
            )
        )
    return chunks


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        loc = f"p.{c.page}" if c.page else "abstract"
        lines.append(f"[{i}] ({c.title} — {c.arxiv_id}, {loc})\n{c.text}")
    return "\n\n".join(lines)
