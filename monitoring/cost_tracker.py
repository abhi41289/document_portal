"""
Token Cost Tracker for LLM API usage.

JD Requirements covered:
  - LLM testing: response validation
  - Model drift detection (cost anomaly = workload change)
  - Alert thresholds with console warnings

Tracks estimated token consumption and USD cost per evaluation run.
Useful for budget governance and detecting abnormal token usage that
may signal prompt injection or runaway generation.

INTEGRATION POINT: Swap _estimate_tokens() with actual token counts
from the LangChain callback handler (LLMCallbackHandler) when running
against the live DocumentPortal API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


# Approximate pricing per 1K tokens (as of mid-2025 for major providers)
_PRICE_PER_1K: dict[str, dict[str, float]] = {
    "groq/llama3-8b":      {"input": 0.0001, "output": 0.0001},
    "groq/mixtral-8x7b":   {"input": 0.0002, "output": 0.0002},
    "gemini-pro":           {"input": 0.00025, "output": 0.0005},
    "gpt-3.5-turbo":        {"input": 0.0005, "output": 0.0015},
    "gpt-4o":               {"input": 0.005,  "output": 0.015},
    "claude-haiku":         {"input": 0.00025, "output": 0.00125},
    "mock":                 {"input": 0.0,    "output": 0.0},
}


@dataclass
class CostSnapshot:
    run_id: str
    timestamp: str
    model: str
    n_queries: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    avg_tokens_per_query: float
    cost_delta_pct: float     # % change vs previous run
    alert: bool


class CostTracker:
    """
    Estimates token usage and USD cost for RAG evaluation runs.

    Usage:
        tracker = CostTracker(model="mock")
        snap = tracker.record_run(questions, answers, run_id="run_001")
        tracker.print_status(snap)
    """

    COST_SPIKE_THRESHOLD = 0.50   # 50% cost increase triggers alert

    def __init__(
        self,
        model: str = "mock",
        log_path: Optional[str] = None,
        monthly_budget_usd: float = 10.0,
    ):
        self.model = model
        self.monthly_budget_usd = monthly_budget_usd
        self.log_path = log_path or os.path.join(
            os.path.dirname(__file__), "cost_log.json"
        )
        self._snapshots: list[CostSnapshot] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_run(
        self,
        questions: list[str],
        answers: list[str],
        contexts: Optional[list[str]] = None,
        run_id: Optional[str] = None,
    ) -> CostSnapshot:
        """
        Estimate and record token costs for a batch of Q&A pairs.

        INTEGRATION POINT: Pass actual token counts from LangChain callbacks
        instead of relying on _estimate_tokens() heuristic.
        """
        run_id = run_id or f"run_{len(self._snapshots) + 1:04d}"
        contexts = contexts or [""] * len(questions)

        input_tokens = 0
        output_tokens = 0

        for q, a, ctx in zip(questions, answers, contexts):
            input_tokens += self._estimate_tokens(q) + self._estimate_tokens(ctx)
            output_tokens += self._estimate_tokens(a)

        cost = self._compute_cost(input_tokens, output_tokens)
        avg_tokens = (input_tokens + output_tokens) / max(len(questions), 1)

        prev_cost = self._snapshots[-1].estimated_cost_usd if self._snapshots else 0.0
        cost_delta_pct = (
            (cost - prev_cost) / prev_cost * 100 if prev_cost > 0 else 0.0
        )
        alert = cost_delta_pct > self.COST_SPIKE_THRESHOLD * 100

        snap = CostSnapshot(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model=self.model,
            n_queries=len(questions),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6),
            avg_tokens_per_query=round(avg_tokens, 1),
            cost_delta_pct=round(cost_delta_pct, 2),
            alert=alert,
        )
        self._snapshots.append(snap)
        self._save()
        return snap

    def print_status(self, snap: CostSnapshot) -> None:
        """Print cost breakdown to console."""
        icon = "🚨" if snap.alert else "💰"
        budget_used = snap.estimated_cost_usd / self.monthly_budget_usd * 100
        total_month = sum(s.estimated_cost_usd for s in self._snapshots)
        budget_pct = total_month / self.monthly_budget_usd * 100

        print(f"\n  {icon} COST TRACKER — {snap.run_id}")
        print(f"     Model:              {snap.model}")
        print(f"     Queries:            {snap.n_queries}")
        print(f"     Input tokens:       {snap.total_input_tokens:,}")
        print(f"     Output tokens:      {snap.total_output_tokens:,}")
        print(f"     Estimated cost:     ${snap.estimated_cost_usd:.6f}")
        print(f"     Avg tokens/query:   {snap.avg_tokens_per_query:.1f}")
        print(f"     Cost Δ vs prev:     {snap.cost_delta_pct:+.1f}%")
        print(f"     Monthly total:      ${total_month:.4f} / ${self.monthly_budget_usd} "
              f"({budget_pct:.1f}% of budget)")

        if snap.alert:
            print(f"\n  ⚠️  ALERT: Cost spike detected ({snap.cost_delta_pct:+.1f}% increase)!")
        if budget_pct > 80:
            print(f"\n  ⚠️  ALERT: Monthly budget {budget_pct:.1f}% consumed!")

    def get_monthly_summary(self) -> dict:
        """Return aggregate cost stats for the current tracked period."""
        total = sum(s.estimated_cost_usd for s in self._snapshots)
        total_tokens = sum(
            s.total_input_tokens + s.total_output_tokens for s in self._snapshots
        )
        return {
            "total_runs": len(self._snapshots),
            "total_cost_usd": round(total, 6),
            "total_tokens": total_tokens,
            "budget_utilisation_pct": round(total / self.monthly_budget_usd * 100, 2),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Approximate token count: ~4 chars per token (GPT-style)."""
        return max(1, len(text) // 4)

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = _PRICE_PER_1K.get(self.model, _PRICE_PER_1K["mock"])
        return (
            input_tokens / 1000 * pricing["input"]
            + output_tokens / 1000 * pricing["output"]
        )

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as fh:
                json.dump([asdict(s) for s in self._snapshots], fh, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._snapshots = [CostSnapshot(**item) for item in raw]
        except Exception:
            self._snapshots = []
