"""Chunk paper text and push it into the Chroma vector store with metadata.

Every chunk is tagged with arxiv_id + title + page number, so later the
citation-checker agent can point back to exactly where a claim came from.
"""
from __future__ import annotations

from typing import Iterable, List

from app.ingestion.fetch_arxiv import ArxivPaper
from app.ingestion.pdf_loader import PageText
from app.rag.vectorstore import get_collection

CHUNK_SIZE = 1200  # characters
CHUNK_OVERLAP = 200


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def index_paper_abstract(paper: ArxivPaper) -> int:
    """Index just the abstract (cheap path, used right after fetch_recent_papers,
    before any PDF is downloaded). Returns number of chunks written."""
    collection = get_collection()
    doc_id = f"{paper.arxiv_id}::abstract"
    collection.upsert(
        ids=[doc_id],
        documents=[paper.abstract],
        metadatas=[
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "source": "abstract",
                "page": 0,
                "pdf_url": paper.pdf_url,
            }
        ],
    )
    return 1


def index_paper_fulltext(paper: ArxivPaper, pages: Iterable[PageText]) -> int:
    """Chunk and index the full extracted text of a downloaded PDF."""
    collection = get_collection()
    ids, documents, metadatas = [], [], []
    for page in pages:
        for j, chunk in enumerate(_chunk_text(page.text)):
            ids.append(f"{paper.arxiv_id}::p{page.page_number}::c{j}")
            documents.append(chunk)
            metadatas.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "source": "fulltext",
                    "page": page.page_number,
                    "pdf_url": paper.pdf_url,
                }
            )
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)
