"""
Benchmark Report Generator — JSON + rich console output.

JD Requirements covered:
  - Benchmark report generator
  - CI/CD integration for continuous AI testing
  - Model drift detection (publishing baseline snapshots)

Outputs:
  1. JSON file: evaluation/reports/benchmark_<timestamp>.json
  2. Console: formatted summary table
  3. Baseline snapshot: evaluation/reports/baseline.json (for drift comparison)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from evaluation.ragas_evaluator import EvaluationReport, SingleResult


REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


class BenchmarkReporter:
    """
    Generates and persists evaluation benchmark reports.

    Usage:
        reporter = BenchmarkReporter()
        reporter.save(report)
        reporter.print_category_breakdown(report)
        reporter.compare_with_baseline(report)
    """

    def __init__(self, reports_dir: str = REPORTS_DIR):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, report: EvaluationReport, filename: Optional[str] = None) -> str:
        """Save report to JSON file. Returns the file path."""
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{ts}.json"

        path = os.path.join(self.reports_dir, filename)
        data = report.to_dict()

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

        print(f"  📄 Report saved → {path}")
        return path

    def save_baseline(self, report: EvaluationReport) -> str:
        """Persist this report as the drift-comparison baseline."""
        path = self.save(report, "baseline.json")
        print(f"  📌 Baseline updated → {path}")
        return path

    def load_baseline(self) -> Optional[dict]:
        """Load the most recent baseline JSON, or None if not found."""
        path = os.path.join(self.reports_dir, "baseline.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_category_breakdown(self, report: EvaluationReport) -> None:
        """Print per-category metric averages."""
        categories: dict[str, list[SingleResult]] = {}
        for r in report.results:
            categories.setdefault(r.category, []).append(r)

        print("\n" + "=" * 72)
        print("  BENCHMARK REPORT — CATEGORY BREAKDOWN")
        print("=" * 72)
        print(
            f"  {'Category':<14} {'Count':>5} {'Faithfulness':>13} "
            f"{'Factuality':>11} {'HallRate':>9} {'Pass%':>6}"
        )
        print("  " + "-" * 68)

        for cat, items in sorted(categories.items()):
            n = len(items)
            avg_faith = sum(r.faithfulness for r in items) / n
            avg_fact = sum(r.factuality for r in items) / n
            hall_rate = sum(1 for r in items if r.hallucination_detected) / n
            pass_pct = sum(1 for r in items if r.passed_gate) / n * 100
            print(
                f"  {cat:<14} {n:>5} {avg_faith:>13.3f} {avg_fact:>11.3f} "
                f"{hall_rate:>9.3f} {pass_pct:>5.1f}%"
            )

        print("  " + "-" * 68)
        print(
            f"  {'TOTAL':<14} {report.total_cases:>5} "
            f"{report.avg_faithfulness:>13.3f} {report.avg_factuality:>11.3f} "
            f"{report.hallucination_rate:>9.3f} "
            f"{report.passed / report.total_cases * 100:>5.1f}%"
        )
        print("=" * 72 + "\n")

    def print_failed_cases(self, report: EvaluationReport) -> None:
        """Print details of failed test cases for debugging."""
        failed = [r for r in report.results if not r.passed_gate]
        if not failed:
            print("  ✅ No failed test cases.\n")
            return

        print(f"\n  FAILED CASES ({len(failed)})")
        print("  " + "-" * 60)
        for r in failed:
            hall = " [HALLUCINATION]" if r.hallucination_detected else ""
            print(f"\n  [{r.test_case_id}]{hall}")
            print(f"  Q:  {r.question[:80]}")
            print(f"  A:  {r.answer[:100]}")
            print(f"  GT: {r.ground_truth[:80]}")
            print(
                f"  Metrics: faith={r.faithfulness:.2f} "
                f"fact={r.factuality:.2f} rel={r.answer_relevance:.2f}"
            )

    # ------------------------------------------------------------------
    # Drift comparison
    # ------------------------------------------------------------------

    def compare_with_baseline(self, report: EvaluationReport) -> dict:
        """
        Compare current report metrics against the saved baseline.

        Returns a dict of metric → delta. Positive = improvement.
        """
        baseline = self.load_baseline()
        if baseline is None:
            print("  ℹ️  No baseline found — saving current report as baseline.")
            self.save_baseline(report)
            return {}

        metrics = [
            "avg_faithfulness", "avg_answer_relevance", "avg_context_relevance",
            "avg_factuality", "hallucination_rate",
        ]
        deltas = {}
        print("\n  DRIFT COMPARISON vs BASELINE")
        print("  " + "-" * 50)
        for m in metrics:
            current = getattr(report, m, 0)
            prev = baseline.get(m, 0)
            delta = current - prev
            # For hallucination_rate lower is better, flip arrow
            arrow = _trend_arrow(delta, lower_is_better=(m == "hallucination_rate"))
            deltas[m] = round(delta, 4)
            print(f"  {m:<35} {prev:.3f} → {current:.3f}  {arrow} {abs(delta):.4f}")

        print("  " + "-" * 50 + "\n")
        return deltas


def _trend_arrow(delta: float, lower_is_better: bool = False) -> str:
    if abs(delta) < 0.001:
        return "→"
    improving = delta > 0 if not lower_is_better else delta < 0
    return "📈" if improving else "📉"
