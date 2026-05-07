"""
Quality Gates — CI/CD build failure logic for RAG evaluation.

JD Requirements covered:
  - Quality gates that can fail a CI/CD build
  - CI/CD integration for continuous AI testing
  - Model drift detection (threshold enforcement)

Usage:
    gate = QualityGate()
    exit_code = gate.enforce(report)  # exits with code 1 if gate fails

The enforce() method is called by the GitHub Actions workflow
(ci_cd_extension/ai_quality_gate.yml) and propagates a non-zero
exit code to fail the PR check.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from evaluation.ragas_evaluator import EvaluationReport


@dataclass
class GateThreshold:
    metric: str
    threshold: float
    direction: str  # "min" (metric must be ≥ threshold) or "max" (metric must be ≤ threshold)
    description: str


class QualityGate:
    """
    Enforces minimum/maximum thresholds on evaluation report metrics.

    Thresholds are tuned for an HR-domain RAG with strict factuality needs.
    Adjust THRESHOLDS dict for different risk profiles.
    """

    THRESHOLDS: list[GateThreshold] = [
        GateThreshold(
            "avg_faithfulness", 0.80, "min",
            "Answers must be ≥80% grounded in retrieved context",
        ),
        GateThreshold(
            "avg_factuality", 0.80, "min",
            "Answers must match ≥80% of expected factual keywords",
        ),
        GateThreshold(
            "hallucination_rate", 0.05, "max",
            "No more than 5% of answers may contain hallucinated content",
        ),
        GateThreshold(
            "avg_answer_relevance", 0.70, "min",
            "Answers must be ≥70% relevant to the question asked",
        ),
        GateThreshold(
            "out_of_scope_refusal_rate", 0.80, "min",
            "≥80% of out-of-scope questions must be correctly refused",
        ),
    ]

    def __init__(self, thresholds: list[GateThreshold] | None = None):
        self._thresholds = thresholds or self.THRESHOLDS

    def check(self, report: EvaluationReport) -> list[dict]:
        """
        Evaluate all thresholds against the report.

        Returns:
            List of violation dicts (empty = all passed).
        """
        violations = []
        for gate in self._thresholds:
            actual = getattr(report, gate.metric, None)
            if actual is None:
                continue

            if gate.direction == "min" and actual < gate.threshold:
                violations.append({
                    "metric": gate.metric,
                    "threshold": gate.threshold,
                    "actual": actual,
                    "direction": gate.direction,
                    "description": gate.description,
                    "gap": round(gate.threshold - actual, 4),
                })
            elif gate.direction == "max" and actual > gate.threshold:
                violations.append({
                    "metric": gate.metric,
                    "threshold": gate.threshold,
                    "actual": actual,
                    "direction": gate.direction,
                    "description": gate.description,
                    "gap": round(actual - gate.threshold, 4),
                })

        return violations

    def enforce(self, report: EvaluationReport, exit_on_fail: bool = False) -> int:
        """
        Print gate results and return exit code (0 = pass, 1 = fail).

        Args:
            report: EvaluationReport from RAGASEvaluator.
            exit_on_fail: If True, calls sys.exit(1) — use in CI pipelines.

        Returns:
            0 if all gates pass, 1 if any gate fails.
        """
        violations = self.check(report)

        print("\n" + "─" * 55)
        print("  QUALITY GATE EVALUATION")
        print("─" * 55)

        all_passed = True
        for gate in self._thresholds:
            actual = getattr(report, gate.metric, None)
            if actual is None:
                continue
            violation = next((v for v in violations if v["metric"] == gate.metric), None)
            if violation:
                all_passed = False
                direction_str = "≥" if gate.direction == "min" else "≤"
                print(
                    f"  ❌ {gate.metric:<35} actual={actual:.3f}  "
                    f"required {direction_str} {gate.threshold:.2f}  gap={violation['gap']:.4f}"
                )
            else:
                direction_str = "≥" if gate.direction == "min" else "≤"
                print(
                    f"  ✅ {gate.metric:<35} actual={actual:.3f}  "
                    f"required {direction_str} {gate.threshold:.2f}"
                )

        print("─" * 55)
        if all_passed:
            print("  🎉 ALL QUALITY GATES PASSED — build continues")
        else:
            print(f"  🚨 {len(violations)} GATE(S) FAILED — build should be blocked")
            for v in violations:
                print(f"     ↳ {v['description']}  (gap={v['gap']:.4f})")

        print("─" * 55 + "\n")

        exit_code = 0 if all_passed else 1
        if exit_on_fail and exit_code != 0:
            sys.exit(exit_code)
        return exit_code

    def generate_github_comment(self, report: EvaluationReport) -> str:
        """
        Generate a Markdown comment for GitHub PR annotation.

        Used by the AI quality gate workflow to post results on PRs.
        """
        violations = self.check(report)
        gate_icon = "✅" if not violations else "❌"
        pct = report.passed / report.total_cases * 100 if report.total_cases else 0

        lines = [
            f"## {gate_icon} AI Quality Gate Report",
            "",
            "| Metric | Value | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ]

        for gate in self._thresholds:
            actual = getattr(report, gate.metric, None)
            if actual is None:
                continue
            violation = next((v for v in violations if v["metric"] == gate.metric), None)
            status = "✅" if not violation else "❌"
            dir_sym = "≥" if gate.direction == "min" else "≤"
            lines.append(
                f"| `{gate.metric}` | `{actual:.3f}` | `{dir_sym}{gate.threshold}` | {status} |"
            )

        lines += [
            "",
            f"**Test Cases:** {report.total_cases} total · {report.passed} passed · {report.failed} failed ({pct:.1f}%)",
            f"**Avg Latency:** {report.avg_latency_ms:.1f} ms",
            f"**Hallucination Rate:** {report.hallucination_rate:.3f}",
            f"**OOS Refusal Rate:** {report.out_of_scope_refusal_rate:.3f}",
            "",
        ]

        if violations:
            lines += [
                "### ❌ Gate Violations",
                "",
            ]
            for v in violations:
                lines.append(f"- **{v['metric']}**: {v['description']} (gap: {v['gap']:.4f})")
            lines += ["", "_Fix these issues before merging._"]
        else:
            lines.append("_All quality gates passed. Safe to merge._")

        return "\n".join(lines)
