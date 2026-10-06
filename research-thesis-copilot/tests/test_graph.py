"""Smoke tests that don't require a live Anthropic API key or a downloaded
embedding model: they monkeypatch `app.agents.nodes.retrieve` (the RAG call)
and `app.agents.nodes.complete` (the LLM call), so the graph's control flow
(retrieve -> draft -> citation_check -> revise/finalize) is exercised
deterministically and fully offline."""
from __future__ import annotations

from unittest.mock import patch

from app.agents import graph as graph_module
from app.rag.retriever import RetrievedChunk


def fake_retrieve(query, k=6, arxiv_id=None):
    return [
        RetrievedChunk(
            text="This paper proposes a retrieval-augmented method for long-context summarization.",
            arxiv_id="2401.00001",
            title="Retrieval-Augmented Long-Context Summarization",
            page=0,
            pdf_url="https://arxiv.org/abs/2401.00001",
            distance=0.1,
        )
    ]


def fake_complete(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    if "citation auditor" in system:
        return "PASS"
    return "This paper proposes retrieval-augmented summarization [1]."


@patch("app.agents.nodes.retrieve", side_effect=fake_retrieve)
@patch("app.agents.nodes.complete", side_effect=fake_complete)
def test_citation_qa_passes_first_try(mock_complete, mock_retrieve):
    result = graph_module.run(
        question="What method does this paper propose?",
        mode="citation_qa",
        arxiv_id="2401.00001",
    )
    assert result["citations_ok"] is True
    assert "[1]" in result["final_answer"]
    assert result["revisions"] == 1


@patch("app.agents.nodes.retrieve", side_effect=fake_retrieve)
def test_citation_check_triggers_one_revision_then_finalizes(mock_retrieve):
    calls = {"n": 0}

    def flaky_complete(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        if "citation auditor" in system:
            calls["n"] += 1
            return "PASS" if calls["n"] > 1 else "FAIL: citation [1] not supported"
        return "Draft referencing [1]."

    with patch("app.agents.nodes.complete", side_effect=flaky_complete):
        result = graph_module.run(
            question="What method does this paper propose?",
            mode="citation_qa",
            arxiv_id="2401.00001",
        )
    assert result["revisions"] == 2
    assert result["citations_ok"] is True


@patch("app.agents.nodes.retrieve", side_effect=fake_retrieve)
def test_exhausting_revisions_still_finalizes_with_a_flag(mock_retrieve):
    def always_fail_complete(system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        if "citation auditor" in system:
            return "FAIL: still unsupported"
        return "Draft referencing [1]."

    with patch("app.agents.nodes.complete", side_effect=always_fail_complete):
        result = graph_module.run(
            question="What method does this paper propose?",
            mode="citation_qa",
            arxiv_id="2401.00001",
        )
    assert result["citations_ok"] is False
    assert result["revisions"] == 2  # MAX_REVISIONS
    assert "could not fully verify" in result["final_answer"]
