"""
Hallucination Rate Trend Tracker.

JD Requirements covered:
  - LLM testing: response validation over time
  - Model drift detection
  - Alert thresholds with console warnings

Tracks per-run hallucination rates and surfaces trend signals so that
a sustained rise triggers early CI intervention before it hits the main
quality gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass
class HallucinationSnapshot:
    run_id: str
    timestamp: str
    total_responses: int
    hallucinated_count: int
    hallucination_rate: float
    categories: dict       # {category: rate}
    trend_delta: float     # current_rate - previous_rate
    alert: bool


class HallucinationTracker:
    """
    Maintains a rolling log of hallucination rates across evaluation runs.

    Features:
      - Per-category tracking (factual, reasoning, edge_case, OOS)
      - 7-run rolling average for trend smoothing
      - Alert when rate exceeds threshold OR shows upward trend ≥ 3 runs
      - JSON persistence for cross-session continuity

    JD: "Hallucination rate trending" + "Alert thresholds with console warnings"
    """

    ALERT_THRESHOLD = 0.05   # absolute rate threshold
    TREND_WINDOW = 3          # consecutive runs showing increase triggers alert

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.path.join(
            os.path.dirname(__file__), "hallucination_log.json"
        )
        self._snapshots: list[HallucinationSnapshot] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        results: list,          # list of SingleResult from evaluation layer
        run_id: Optional[str] = None,
    ) -> HallucinationSnapshot:
        """
        Record hallucination statistics from an evaluation run.

        Args:
            results: List of SingleResult objects (from RAGASEvaluator).
            run_id: Optional run identifier; auto-generated if omitted.
        """
        run_id = run_id or f"run_{len(self._snapshots) + 1:04d}"
        total = len(results)
        hallucinated = sum(1 for r in results if r.hallucination_detected)
        rate = hallucinated / total if total else 0.0

        # Per-category rates
        categories: dict[str, dict] = {}
        for r in results:
            cat = r.category
            if cat not in categories:
                categories[cat] = {"total": 0, "hallucinated": 0}
            categories[cat]["total"] += 1
            if r.hallucination_detected:
                categories[cat]["hallucinated"] += 1

        cat_rates = {
            cat: round(v["hallucinated"] / v["total"], 4)
            for cat, v in categories.items()
            if v["total"] > 0
        }

        prev_rate = self._snapshots[-1].hallucination_rate if self._snapshots else 0.0
        trend_delta = round(rate - prev_rate, 4)
        alert = rate > self.ALERT_THRESHOLD or self._sustained_upward_trend()

        snap = HallucinationSnapshot(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_responses=total,
            hallucinated_count=hallucinated,
            hallucination_rate=round(rate, 4),
            categories=cat_rates,
            trend_delta=trend_delta,
            alert=alert,
        )
        self._snapshots.append(snap)
        self._save()
        return snap

    def print_status(self, snap: HallucinationSnapshot) -> None:
        """Print a formatted hallucination status report."""
        icon = "🚨" if snap.alert else "✅"
        trend = f"{snap.trend_delta:+.4f}" if snap.trend_delta else "  N/A"
        print(f"\n  {icon} HALLUCINATION TRACKER — {snap.run_id}")
        print(f"     Rate:          {snap.hallucination_rate:.4f}  (threshold ≤ {self.ALERT_THRESHOLD})")
        print(f"     Count:         {snap.hallucinated_count} / {snap.total_responses}")
        print(f"     Trend Δ:       {trend}")
        print("     By category:")
        for cat, rate in snap.categories.items():
            flag = " ⚠️" if rate > self.ALERT_THRESHOLD else ""
            print(f"       {cat:<16} {rate:.4f}{flag}")
        if snap.alert:
            print("\n  ⚠️  ALERT: Hallucination rate exceeded threshold or shows upward trend!")

    def print_trend_chart(self, last_n: int = 8) -> None:
        """Print an ASCII sparkline of recent hallucination rates."""
        history = self._snapshots[-last_n:]
        if not history:
            return

        print(f"\n  HALLUCINATION TREND (last {len(history)} runs)")
        print("  " + "-" * 48)
        max_rate = max(s.hallucination_rate for s in history) or 0.10
        bar_width = 30

        for snap in history:
            fill = int(snap.hallucination_rate / max_rate * bar_width)
            bar = "█" * fill + "░" * (bar_width - fill)
            alert_flag = " 🚨" if snap.alert else ""
            print(
                f"  {snap.run_id:<12} [{bar}] {snap.hallucination_rate:.4f}{alert_flag}"
            )
        print(f"  {'':12}  0{'':>{bar_width - 2}}{max_rate:.4f}")

    def rolling_average(self, window: int = 7) -> float:
        """Return the rolling average hallucination rate over the last N runs."""
        recent = self._snapshots[-window:]
        if not recent:
            return 0.0
        return sum(s.hallucination_rate for s in recent) / len(recent)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sustained_upward_trend(self) -> bool:
        """True if the last TREND_WINDOW runs show consecutive rate increases."""
        if len(self._snapshots) < self.TREND_WINDOW:
            return False
        recent = self._snapshots[-self.TREND_WINDOW:]
        return all(
            recent[i].hallucination_rate > recent[i - 1].hallucination_rate
            for i in range(1, len(recent))
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
            self._snapshots = [HallucinationSnapshot(**item) for item in raw]
        except Exception:
            self._snapshots = []
