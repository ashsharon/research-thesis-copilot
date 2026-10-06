"""The individual graph nodes. Each is a plain function: (state) -> partial state update.

Pipeline shape:

    retrieve -> draft_{mode} -> citation_check -> (revise draft_{mode})* -> finalize

The citation checker is what makes this "citation-checked QA" rather than a
bare RAG wrapper: it re-reads the draft against the retrieved context and
fails it if it finds a claim the context doesn't support, which routes back
to the drafting node for one revision before giving up and flagging it.
"""
from __future__ import annotations

from app.agents.llm import complete
from app.agents.state import CoPilotState
from app.rag.retriever import format_context, retrieve

MAX_REVISIONS = 2

SYSTEM_PROMPTS = {
    "literature_review": (
        "You are a meticulous research assistant helping an M.Sc. student write a "
        "literature review section. Using ONLY the numbered context snippets given, "
        "synthesize the common themes, methods and findings across the papers. "
        "Every substantive claim must end with a bracketed citation like [2] "
        "referring to the snippet number it came from. Do not invent papers or "
        "results that are not in the context."
    ),
    "gap_analysis": (
        "You are a research advisor. Using ONLY the numbered context snippets given, "
        "identify concrete, specific gaps in the current literature: open problems, "
        "unexplored combinations of methods, contradictory findings, or populations/"
        "datasets that are under-studied. For each gap, cite the snippet(s) [n] that "
        "justify why it is actually a gap rather than something already solved. "
        "End with 2-3 candidate thesis research questions."
    ),
    "citation_qa": (
        "You are answering a direct factual question about a specific paper. Using "
        "ONLY the numbered context snippets given, answer precisely and concisely. "
        "Every sentence must carry a bracketed citation [n]. If the context does not "
        "contain the answer, say so explicitly instead of guessing."
    ),
}


def retrieve_node(state: CoPilotState) -> dict:
    chunks = retrieve(state["question"], k=8, arxiv_id=state.get("arxiv_id"))
    return {
        "context": format_context(chunks),
        "sources": [
            {"title": c.title, "arxiv_id": c.arxiv_id, "page": c.page, "pdf_url": c.pdf_url}
            for c in chunks
        ],
        "revisions": 0,
    }


def draft_node(state: CoPilotState) -> dict:
    system = SYSTEM_PROMPTS[state["mode"]]
    critique_note = (
        f"\n\nYour previous draft was rejected by the citation checker for this reason, fix it:\n{state['critique']}"
        if state.get("critique")
        else ""
    )
    user = (
        f"Question / task: {state['question']}\n\n"
        f"Context snippets:\n{state['context']}"
        f"{critique_note}"
    )
    draft = complete(system=system, user=user, max_tokens=1600)
    return {"draft": draft}


def citation_check_node(state: CoPilotState) -> dict:
    system = (
        "You are a strict citation auditor. You will be given a context (numbered "
        "snippets) and a draft that claims to be grounded in that context. Check "
        "whether every citation number used in the draft actually appears in the "
        "context, and whether the sentence it's attached to is actually supported "
        "by that snippet (not just topically related). "
        "Reply with exactly one line: either 'PASS' or 'FAIL: <short reason>'."
    )
    user = f"Context:\n{state['context']}\n\nDraft:\n{state['draft']}"
    verdict = complete(system=system, user=user, max_tokens=200, temperature=0.0)
    passed = verdict.strip().upper().startswith("PASS")
    return {
        "citations_ok": passed,
        "critique": None if passed else verdict.strip(),
        "revisions": state.get("revisions", 0) + 1,
    }


def finalize_node(state: CoPilotState) -> dict:
    status_note = (
        ""
        if state.get("citations_ok")
        else "\n\n_Note: the citation checker could not fully verify this draft after "
        f"{state.get('revisions', 0)} attempt(s); double-check the flagged claims before using them._"
    )
    return {"final_answer": state["draft"] + status_note}


def should_revise(state: CoPilotState) -> str:
    if state.get("citations_ok"):
        return "finalize"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "finalize"
    return "revise"
