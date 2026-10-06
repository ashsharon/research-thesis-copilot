"""Pull recent papers from arXiv for a given category/query.

This is the thing the n8n "paper ingestion" workflow calls on a schedule
(e.g. every morning) via the FastAPI /ingest/arxiv endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import arxiv

from app.config import settings


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    pdf_url: str
    published: str
    categories: List[str]


def fetch_recent_papers(
    query: str | None = None,
    category: str | None = None,
    max_results: int | None = None,
) -> List[ArxivPaper]:
    """Fetch the most recent arXiv papers matching a query or category.

    Either `query` (free text, searched in title/abstract) or `category`
    (e.g. "cs.CL") can be supplied; if both are given they are ANDed.
    """
    category = category or settings.arxiv_category
    max_results = max_results or settings.arxiv_max_results

    terms = []
    if category:
        terms.append(f"cat:{category}")
    if query:
        terms.append(f"all:{query}")
    search_query = " AND ".join(terms) if terms else f"cat:{settings.arxiv_category}"

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[ArxivPaper] = []
    client = arxiv.Client()
    for result in client.results(search):
        papers.append(
            ArxivPaper(
                arxiv_id=result.get_short_id(),
                title=result.title.strip().replace("\n", " "),
                abstract=result.summary.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                pdf_url=result.pdf_url,
                published=result.published.isoformat(),
                categories=result.categories,
            )
        )
    return papers


def download_pdf(paper: ArxivPaper) -> str:
    """Download a paper's PDF into the local papers directory, return the path."""
    import arxiv as _arxiv  # local import keeps module import light for tests

    client = _arxiv.Client()
    search = _arxiv.Search(id_list=[paper.arxiv_id])
    result = next(client.results(search))
    dest = settings.papers_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    result.download_pdf(dirpath=str(settings.papers_dir), filename=dest.name)
    return str(dest)
