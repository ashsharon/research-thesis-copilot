"""FastAPI surface that n8n (and anything else) talks to.

Endpoints:
  POST /ingest/arxiv        -> fetch + index recent abstracts for a category/query
  POST /ingest/pdf          -> download + fully index one paper's PDF by arxiv_id
  POST /ask                 -> run the LangGraph co-pilot (literature_review / gap_analysis / citation_qa)
  GET  /digest/weekly       -> build this week's digest (new papers + one gap-analysis pass)
  GET  /health
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.graph import run as run_copilot
from app.config import settings
from app.ingestion.chunk_and_embed import index_paper_abstract, index_paper_fulltext
from app.ingestion.fetch_arxiv import download_pdf, fetch_recent_papers
from app.ingestion.pdf_loader import load_pdf_pages

app = FastAPI(title="Research & Thesis Co-Pilot")


class IngestArxivRequest(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    max_results: Optional[int] = None


class IngestPdfRequest(BaseModel):
    arxiv_id: str


class AskRequest(BaseModel):
    question: str
    mode: str = "citation_qa"  # literature_review | gap_analysis | citation_qa
    arxiv_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.claude_model}


@app.post("/ingest/arxiv")
def ingest_arxiv(req: IngestArxivRequest):
    papers = fetch_recent_papers(query=req.query, category=req.category, max_results=req.max_results)
    indexed = sum(index_paper_abstract(p) for p in papers)
    return {
        "fetched": len(papers),
        "chunks_indexed": indexed,
        "papers": [{"arxiv_id": p.arxiv_id, "title": p.title} for p in papers],
    }


@app.post("/ingest/pdf")
def ingest_pdf(req: IngestPdfRequest):
    papers = fetch_recent_papers(query=f'id:"{req.arxiv_id}"', max_results=1)
    if not papers:
        raise HTTPException(status_code=404, detail="arxiv_id not found")
    paper = papers[0]
    path = download_pdf(paper)
    pages = load_pdf_pages(path)
    n = index_paper_fulltext(paper, pages)
    return {"arxiv_id": paper.arxiv_id, "pages_indexed": len(pages), "chunks_indexed": n}


@app.post("/ask")
def ask(req: AskRequest):
    if req.mode not in {"literature_review", "gap_analysis", "citation_qa"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    result = run_copilot(question=req.question, mode=req.mode, arxiv_id=req.arxiv_id)
    return {
        "answer": result.get("final_answer"),
        "citations_ok": result.get("citations_ok"),
        "sources": result.get("sources"),
        "revisions": result.get("revisions"),
    }


@app.get("/digest/weekly")
def weekly_digest():
    """Pulls this week's new papers for the configured category and runs one
    gap-analysis pass over the pooled abstracts — this is what the n8n
    weekly-digest workflow emails out."""
    papers = fetch_recent_papers(max_results=settings.arxiv_max_results)
    for p in papers:
        index_paper_abstract(p)

    titles_block = "\n".join(f"- {p.title} ({p.arxiv_id})" for p in papers)
    gap = run_copilot(
        question=(
            "Based on the newly indexed abstracts below, what are the most "
            "interesting emerging directions this week, and are there any that "
            "connect to an under-explored angle?\n\n" + titles_block
        ),
        mode="gap_analysis",
    )
    return {
        "new_papers": [{"arxiv_id": p.arxiv_id, "title": p.title, "pdf_url": p.pdf_url} for p in papers],
        "gap_analysis": gap.get("final_answer"),
        "recipient": settings.digest_recipient,
    }


def main():
    import uvicorn

    uvicorn.run("app.api.server:app", host=settings.api_host, port=settings.api_port, reload=True)


if __name__ == "__main__":
    main()
