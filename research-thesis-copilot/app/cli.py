"""Quick command-line entry point, useful before wiring up n8n at all.

Examples:
  python -m app.cli ingest --category cs.CL --max-results 10
  python -m app.cli ask "What methods do these papers use for long-context retrieval?" --mode literature_review
"""
from __future__ import annotations

import argparse
import json

from app.agents.graph import run as run_copilot
from app.ingestion.chunk_and_embed import index_paper_abstract
from app.ingestion.fetch_arxiv import fetch_recent_papers


def cmd_ingest(args):
    papers = fetch_recent_papers(query=args.query, category=args.category, max_results=args.max_results)
    for p in papers:
        index_paper_abstract(p)
        print(f"indexed: {p.arxiv_id}  {p.title}")
    print(f"\n{len(papers)} papers indexed.")


def cmd_ask(args):
    result = run_copilot(question=args.question, mode=args.mode, arxiv_id=args.arxiv_id)
    print(json.dumps({
        "answer": result.get("final_answer"),
        "citations_ok": result.get("citations_ok"),
        "sources": result.get("sources"),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Research & Thesis Co-Pilot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--query", default=None)
    p_ingest.add_argument("--category", default=None)
    p_ingest.add_argument("--max-results", type=int, default=None, dest="max_results")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", default="citation_qa",
                        choices=["literature_review", "gap_analysis", "citation_qa"])
    p_ask.add_argument("--arxiv-id", default=None, dest="arxiv_id")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
