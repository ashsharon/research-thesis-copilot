"""Assembles the LangGraph StateGraph:

        ┌────────────┐
        │  retrieve  │
        └─────┬──────┘
              ▼
        ┌────────────┐
   ┌───▶│    draft   │
   │    └─────┬──────┘
   │          ▼
   │  ┌───────────────┐
   │  │ citation_check│
   │  └───────┬───────┘
   │          │ FAIL (revisions < max) -> revise
   └──────────┘
              │ PASS or out of revisions
              ▼
        ┌────────────┐
        │  finalize  │
        └────────────┘

One graph serves all three modes (literature_review / gap_analysis /
citation_qa) — only the system prompt picked in nodes.py changes.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    citation_check_node,
    draft_node,
    finalize_node,
    retrieve_node,
    should_revise,
)
from app.agents.state import CoPilotState


def build_graph():
    graph = StateGraph(CoPilotState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("citation_check", citation_check_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "citation_check")
    graph.add_conditional_edges(
        "citation_check",
        should_revise,
        {"revise": "draft", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import time; FastAPI and the CLI both reuse this.
copilot_graph = build_graph()


def run(question: str, mode: str, arxiv_id: str | None = None) -> dict:
    initial_state: CoPilotState = {"question": question, "mode": mode, "arxiv_id": arxiv_id}
    return copilot_graph.invoke(initial_state)
