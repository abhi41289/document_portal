"""Trace capture helpers for QA evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    stage: str
    started_at: str
    ended_at: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self):
        self.events: List[TraceEvent] = []

    def record(
        self,
        stage: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> TraceEvent:
        start = started_at or datetime.utcnow()
        end = ended_at or datetime.utcnow()
        event = TraceEvent(
            stage=stage,
            started_at=start.isoformat(),
            ended_at=end.isoformat(),
            request=request,
            response=response,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self.events]

    def failure_reasons(self) -> List[str]:
        reasons: List[str] = []
        for event in self.events:
            reason = event.metadata.get("failure_reason")
            if reason:
                reasons.append(str(reason))
        return reasons

