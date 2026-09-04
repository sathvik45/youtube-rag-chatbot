"""Deterministic evidence-gate tests with no Pinecone or LLM calls."""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.graph import nodes
from src.graph.build_graph import build_graph
from src.rag import retriever
from src.rag.evidence import (
    ScoredCandidate,
    candidate_fetch_limit,
    select_evidence,
)


def _candidate(video_id: str, score: float) -> ScoredCandidate[str]:
    return ScoredCandidate(
        value=f"{video_id}:{score}",
        video_id=video_id,
        score=score,
    )


def test_score_gate_keeps_the_equal_boundary_and_caps_after_filtering() -> None:
    selection = select_evidence(
        [
            _candidate("video-a", 0.91),
            _candidate("video-a", 0.80),
            _candidate("video-b", 0.80),
            _candidate("video-b", 0.79),
        ],
        k=2,
        per_video_cap=1,
        single_video=False,
        score_threshold=0.80,
    )

    assert [candidate.score for candidate in selection.qualifying] == [
        0.91,
        0.80,
        0.80,
    ]
    assert [candidate.value for candidate in selection.selected] == [
        "video-a:0.91",
        "video-b:0.8",
    ]


def test_score_gate_drops_all_low_scored_candidates() -> None:
    selection = select_evidence(
        [_candidate("video-a", 0.49)],
        k=5,
        per_video_cap=None,
        single_video=True,
        score_threshold=0.50,
    )

    assert selection.qualifying == ()
    assert selection.selected == ()


def test_unconfigured_gate_preserves_existing_top_k_retrieval() -> None:
    selection = select_evidence(
        [_candidate("video-a", 0.01)],
        k=5,
        per_video_cap=None,
        single_video=True,
        score_threshold=None,
    )

    assert [candidate.value for candidate in selection.selected] == [
        "video-a:0.01"
    ]


def test_playlist_candidate_pool_is_larger_before_the_per_video_cap() -> None:
    assert candidate_fetch_limit(k=5, scope_size=1, per_video_cap=2) == 5
    assert candidate_fetch_limit(k=5, scope_size=3, per_video_cap=2) == 20


class _FakeVectorStore:
    def __init__(self, matches: list[tuple[Document, float]]) -> None:
        self.matches = matches
        self.calls: list[dict] = []

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int,
        filter: dict,
    ) -> list[tuple[Document, float]]:
        self.calls.append({"query": query, "k": k, "filter": filter})
        return self.matches


def test_retrieve_filters_a_low_score_before_the_graph_can_see_it(
    monkeypatch,
    caplog,
) -> None:
    question = "Do they discuss an unrelated topic?"
    store = _FakeVectorStore(
        [
            (
                Document(
                    page_content="Topically similar but insufficient evidence.",
                    metadata={"video_id": "abcdefghijk", "chunk_index": 0},
                ),
                0.49,
            )
        ]
    )
    monkeypatch.setattr(retriever, "get_vectorstore", lambda: store)

    with caplog.at_level("INFO", logger="src.rag.retriever"):
        documents = retriever.retrieve(
            question,
            ["abcdefghijk"],
            k=5,
            score_threshold=0.50,
        )

    assert documents == []
    assert store.calls == [
        {
            "query": question,
            "k": 5,
            "filter": {"video_id": "abcdefghijk"},
        }
    ]
    gate_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.rag.retriever"
        and record.getMessage().startswith("retrieval_evidence_gate ")
    ]
    assert len(gate_logs) == 1
    assert "candidate_count=1" in gate_logs[0]
    assert "qualifying_count=0" in gate_logs[0]
    assert question not in gate_logs[0]


def test_real_graph_refuses_low_score_evidence_without_answer_generation(
    monkeypatch,
) -> None:
    store = _FakeVectorStore(
        [
            (
                Document(
                    page_content="Weak, unrelated evidence.",
                    metadata={"video_id": "abcdefghijk", "chunk_index": 0},
                ),
                0.49,
            )
        ]
    )
    monkeypatch.setattr(retriever, "get_vectorstore", lambda: store)
    monkeypatch.setattr(nodes.settings, "score_threshold", 0.50)

    async def generation_must_not_run(_state: dict) -> dict:
        raise AssertionError("Generation must not run without qualifying evidence.")

    monkeypatch.setattr(nodes, "Generate_node", generation_must_not_run)
    graph = build_graph()

    state = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Ask an unsupported question")]},
            config={"configurable": {"video_ids": ["abcdefghijk"]}},
        )
    )

    assert state["outcome"] == "refused_no_context"
    assert state["grounded"] is None
    assert state["citations"] == []
    assert state["messages"][-1].content == nodes.NO_CONTEXT_MESSAGE


def test_real_graph_generates_when_evidence_meets_the_threshold(
    monkeypatch,
) -> None:
    store = _FakeVectorStore(
        [
            (
                Document(
                    page_content="The supported evidence is here.",
                    metadata={
                        "video_id": "abcdefghijk",
                        "chunk_index": 0,
                        "start_ms": 0,
                        "end_ms": 1_000,
                    },
                ),
                0.50,
            )
        ]
    )
    monkeypatch.setattr(retriever, "get_vectorstore", lambda: store)
    monkeypatch.setattr(nodes.settings, "score_threshold", 0.50)

    generated = False

    async def fake_generate(_state: dict) -> dict:
        nonlocal generated
        generated = True
        return {
            "messages": [AIMessage(content="Supported answer.")],
            "context": "The supported evidence is here.",
        }

    def fake_cite(_state: dict) -> dict:
        return {
            "citations": [],
            "grounded": True,
            "outcome": "answered",
        }

    monkeypatch.setattr(nodes, "Generate_node", fake_generate)
    monkeypatch.setattr(nodes, "cite_node", fake_cite)
    graph = build_graph()

    state = asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Ask a supported question")]},
            config={"configurable": {"video_ids": ["abcdefghijk"]}},
        )
    )

    assert generated is True
    assert state["outcome"] == "answered"
