"""Structured tracing helpers for conversational RAG invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class TraceStage:
    """A single stage in an observed RAG request."""

    name: str
    started_at: str
    ended_at: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.ended_at)
        return max((end - start).total_seconds() * 1000.0, 0.0)


@dataclass
class RAGInvocationTrace:
    """Trace envelope for a single chat request."""

    question: str
    session_id: Optional[str]
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_iso_now)
    ended_at: Optional[str] = None
    status: str = "running"
    failure_reason: Optional[str] = None
    error_message: Optional[str] = None
    stages: List[TraceStage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage(
        self,
        name: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
    ) -> TraceStage:
        stage = TraceStage(
            name=name,
            started_at=started_at or _iso_now(),
            ended_at=ended_at or _iso_now(),
            request=request,
            response=response,
            metadata=metadata or {},
        )
        self.stages.append(stage)
        return stage

    def succeed(self, **metadata: Any) -> None:
        self.status = "success"
        self.ended_at = _iso_now()
        if metadata:
            self.metadata.update(metadata)

    def fail(self, reason: str, error_message: Optional[str] = None, **metadata: Any) -> None:
        self.status = "error"
        self.failure_reason = reason
        self.error_message = error_message
        self.ended_at = _iso_now()
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        started = datetime.fromisoformat(self.started_at)
        ended = datetime.fromisoformat(self.ended_at) if self.ended_at else datetime.utcnow()
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_time_ms": max((ended - started).total_seconds() * 1000.0, 0.0),
            "failure_reason": self.failure_reason,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "stages": [
                {
                    "name": stage.name,
                    "started_at": stage.started_at,
                    "ended_at": stage.ended_at,
                    "duration_ms": stage.duration_ms,
                    "request": stage.request,
                    "response": stage.response,
                    "metadata": stage.metadata,
                }
                for stage in self.stages
            ],
        }

