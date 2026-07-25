"""Observability and diagnostics coverage."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.qa.observability import TraceRecorder


@pytest.mark.observability
def test_trace_recorder_captures_pipeline_artifacts(trace_recorder):
    started = datetime.utcnow()
    ended = started + timedelta(milliseconds=15)

    trace_recorder.record(
        stage="retrieval",
        request={"query": "Who is the CEO?"},
        response={
            "retrieved_docs": ["doc_company"],
            "context": "Alice Johnson is the CEO of TechCorp.",
        },
        metadata={"top_k": 3},
        started_at=started,
        ended_at=ended,
    )
    trace_recorder.record(
        stage="generation",
        request={"question": "Who is the CEO?"},
        response={"answer": "Alice Johnson is the CEO of TechCorp."},
        metadata={"prompt_version": "v2_enhanced"},
        started_at=started,
        ended_at=ended,
    )

    payload = trace_recorder.to_dict()

    assert len(payload) == 2
    assert payload[0]["stage"] == "retrieval"
    assert payload[1]["response"]["answer"] == "Alice Johnson is the CEO of TechCorp."


@pytest.mark.observability
def test_trace_recorder_failure_reason_summary():
    recorder = TraceRecorder()
    recorder.record(
        stage="sql",
        request={"query": "DROP TABLE customers;"},
        response={"refusal": True},
        metadata={"failure_reason": "unsafe_sql"},
    )
    recorder.record(
        stage="execution",
        request={"query": "SELECT 1"},
        response={"rows": [{"1": 1}]},
        metadata={"failure_reason": "execution_timeout"},
    )

    assert recorder.failure_reasons() == ["unsafe_sql", "execution_timeout"]

