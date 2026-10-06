"""Shared state object passed between every node in the LangGraph graph."""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

Mode = Literal["literature_review", "gap_analysis", "citation_qa"]


class CoPilotState(TypedDict, total=False):
    # input
    question: str
    mode: Mode
    arxiv_id: Optional[str]  # restrict RAG to one paper, e.g. for citation_qa

    # working memory
    context: str                 # formatted retrieved chunks
    sources: List[dict]          # raw chunk metadata, for the final citation list
    draft: str                   # the agent's current draft answer
    critique: Optional[str]      # feedback from the citation checker
    revisions: int                # how many times we've looped draft -> check

    # output
    final_answer: str
    citations_ok: bool
