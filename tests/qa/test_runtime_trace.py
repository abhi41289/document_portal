"""Runtime trace tests for conversational RAG."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.document_chat.retrieval import ConversationalRAG
from exception.custom_exception import DocumentPortalException


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, prompt):
        content = self.responses.pop(0) if self.responses else "Not available in provided documents."
        return SimpleNamespace(content=content)


class DummyRetriever:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def invoke(self, query):
        if self.should_fail:
            raise ValueError("retrieval failed")
        return [
            SimpleNamespace(
                page_content="Alice Johnson is the CEO of TechCorp.",
                metadata={"source_document_id": "doc_company"},
            )
        ]


@pytest.mark.observability
def test_conversational_rag_returns_trace(monkeypatch):
    monkeypatch.setattr(
        "src.document_chat.retrieval.ModelLoader",
        lambda: SimpleNamespace(load_llm=lambda: DummyLLM(
            [
                "Who is the CEO of TechCorp?",
                "Alice Johnson is the CEO of TechCorp.",
            ]
        )),
    )

    rag = ConversationalRAG(session_id="trace-session")
    rag.retriever = DummyRetriever()

    answer, trace = rag.invoke_with_trace("Who is the CEO?", chat_history=[])

    assert answer == "Alice Johnson is the CEO of TechCorp."
    assert trace["status"] == "success"
    assert trace["trace_id"]
    assert len(trace["stages"]) == 3
    assert trace["stages"][0]["name"] == "rewrite"
    assert trace["stages"][1]["response"]["retrieved_documents"][0]["metadata"]["source_document_id"] == "doc_company"
    assert trace["stages"][2]["response"]["answer"] == "Alice Johnson is the CEO of TechCorp."


@pytest.mark.observability
def test_conversational_rag_attaches_trace_to_exceptions(monkeypatch):
    monkeypatch.setattr(
        "src.document_chat.retrieval.ModelLoader",
        lambda: SimpleNamespace(load_llm=lambda: DummyLLM(["standalone question"])),
    )

    rag = ConversationalRAG(session_id="trace-session")
    rag.retriever = DummyRetriever(should_fail=True)

    with pytest.raises(DocumentPortalException) as exc_info:
        rag.invoke_with_trace("Who is the CEO?", chat_history=[])

    trace = getattr(exc_info.value, "trace", None)
    assert trace is not None
    assert trace["status"] == "error"
    assert trace["failure_reason"] == "valueerror"

