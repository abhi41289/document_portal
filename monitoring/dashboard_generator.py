"""
HTML Dashboard Generator for RAG monitoring results.

JD Requirements covered:
  - HTML dashboard report generator
  - Model drift detection visualisation
  - CI/CD integration for continuous AI testing

Generates a self-contained HTML file with:
  - Evaluation summary KPI cards
  - Drift trend chart (SVG sparkline — no external dependencies)
  - Hallucination rate bar chart
  - Cost breakdown table
  - Failed test case details

Output: monitoring/dashboard/rag_monitor_<timestamp>.html
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from evaluation.ragas_evaluator import EvaluationReport
from monitoring.evidently_monitor import DriftMetrics
from monitoring.hallucination_tracker import HallucinationSnapshot
from monitoring.cost_tracker import CostSnapshot


DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


class DashboardGenerator:
    """
    Renders a self-contained HTML monitoring dashboard.

    Requires zero external JS/CSS dependencies — all styling is inline
    so the file renders correctly in GitHub Actions artefacts and email.
    """

    def __init__(self, output_dir: str = DASHBOARD_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        eval_report: EvaluationReport,
        drift_metrics: Optional[DriftMetrics] = None,
        hall_snapshot: Optional[HallucinationSnapshot] = None,
        cost_snapshot: Optional[CostSnapshot] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Build and save the HTML dashboard.

        Returns:
            Absolute path to the saved HTML file.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_monitor_{ts}.html"

        path = os.path.join(self.output_dir, filename)
        html = self._render(eval_report, drift_metrics, hall_snapshot, cost_snapshot)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)

        print(f"  🌐 Dashboard saved → {path}")
        return path

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        report: EvaluationReport,
        drift: Optional[DriftMetrics],
        hall: Optional[HallucinationSnapshot],
        cost: Optional[CostSnapshot],
    ) -> str:
        gate_color = "#27ae60" if report.gate_passed else "#e74c3c"
        gate_text = "PASSED ✅" if report.gate_passed else "FAILED ❌"
        pass_pct = round(report.passed / report.total_cases * 100, 1) if report.total_cases else 0

        kpi_cards = self._kpi_cards(report, drift, hall, cost, gate_color, gate_text)
        metrics_table = self._metrics_table(report)
        failed_table = self._failed_cases_table(report)
        drift_section = self._drift_section(drift) if drift else ""
        hall_section = self._hall_section(hall) if hall else ""
        cost_section = self._cost_section(cost) if cost else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Monitor Dashboard — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f4f6f9; color: #2c3e50; font-size: 14px; }}
  .header {{ background: #2c3e50; color: white; padding: 20px 30px;
             display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 22px; }}
  .header .subtitle {{ font-size: 12px; opacity: 0.7; }}
  .gate-badge {{ background: {gate_color}; color: white; padding: 8px 18px;
                 border-radius: 20px; font-weight: bold; font-size: 15px; }}
  .container {{ max-width: 1200px; margin: 24px auto; padding: 0 20px; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
              gap: 16px; margin-bottom: 24px; }}
  .kpi-card {{ background: white; border-radius: 8px; padding: 16px 20px;
               box-shadow: 0 2px 6px rgba(0,0,0,.08); }}
  .kpi-card .value {{ font-size: 28px; font-weight: 700; }}
  .kpi-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; }}
  .kpi-card.good {{ border-left: 4px solid #27ae60; }}
  .kpi-card.warn {{ border-left: 4px solid #f39c12; }}
  .kpi-card.bad  {{ border-left: 4px solid #e74c3c; }}
  .section {{ background: white; border-radius: 8px; padding: 20px 24px;
              box-shadow: 0 2px 6px rgba(0,0,0,.08); margin-bottom: 20px; }}
  .section h2 {{ font-size: 16px; margin-bottom: 14px; border-bottom: 1px solid #ecf0f1;
                 padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fa; text-align: left; padding: 8px 12px;
        border-bottom: 2px solid #dee2e6; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #f8f9fa; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 600; }}
  .pass {{ background: #d5f5e3; color: #196f3d; }}
  .fail {{ background: #fadbd8; color: #922b21; }}
  .warn-b {{ background: #fdebd0; color: #935116; }}
  .bar-bg {{ background: #ecf0f1; border-radius: 4px; height: 8px; }}
  .bar-fill {{ height: 8px; border-radius: 4px; background: #3498db; }}
  .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🔍 RAG Quality Monitor Dashboard</h1>
    <div class="subtitle">DocumentPortal · AI QA POC · {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
  </div>
  <div class="gate-badge">Quality Gate: {gate_text}</div>
</div>

<div class="container">

  <div class="kpi-row">
    {kpi_cards}
  </div>

  <div class="section">
    <h2>📊 Evaluation Metrics Summary</h2>
    {metrics_table}
  </div>

  {drift_section}
  {hall_section}
  {cost_section}

  <div class="section">
    <h2>❌ Failed Test Cases ({report.failed})</h2>
    {failed_table}
  </div>

</div>

<div class="footer">
  Generated by RAG Monitor POC · Intellectt AI QA Demo ·
  Model: {report.model_id} · Cases: {report.total_cases}
</div>
</body>
</html>"""

    def _kpi_cards(self, report, drift, hall, cost, gate_color, gate_text):
        pass_pct = round(report.passed / report.total_cases * 100, 1) if report.total_cases else 0
        hall_rate = report.hallucination_rate
        faith = report.avg_faithfulness

        faith_cls = "good" if faith >= 0.80 else ("warn" if faith >= 0.65 else "bad")
        hall_cls = "good" if hall_rate <= 0.05 else ("warn" if hall_rate <= 0.10 else "bad")
        pass_cls = "good" if pass_pct >= 90 else ("warn" if pass_pct >= 70 else "bad")

        cards = [
            (f"{pass_pct}%", "Test Pass Rate", pass_cls),
            (f"{faith:.3f}", "Avg Faithfulness", faith_cls),
            (f"{report.avg_factuality:.3f}", "Avg Factuality",
             "good" if report.avg_factuality >= 0.80 else "warn"),
            (f"{hall_rate:.3f}", "Hallucination Rate", hall_cls),
            (f"{report.out_of_scope_refusal_rate:.3f}", "OOS Refusal Rate",
             "good" if report.out_of_scope_refusal_rate >= 0.80 else "warn"),
            (f"{report.avg_latency_ms:.0f} ms", "Avg Latency", "good"),
        ]
        if drift:
            drift_cls = "good" if not drift.drift_detected else (
                "warn" if drift.alert_level == "warning" else "bad"
            )
            cards.append((f"{drift.embedding_psi:.3f}", "Embedding PSI", drift_cls))

        html = ""
        for value, label, cls in cards:
            html += f"""
    <div class="kpi-card {cls}">
      <div class="value">{value}</div>
      <div class="label">{label}</div>
    </div>"""
        return html

    def _metrics_table(self, report: EvaluationReport) -> str:
        rows = [
            ("avg_faithfulness", report.avg_faithfulness, 0.80, True),
            ("avg_answer_relevance", report.avg_answer_relevance, 0.70, True),
            ("avg_context_relevance", report.avg_context_relevance, 0.70, True),
            ("avg_factuality", report.avg_factuality, 0.80, True),
            ("hallucination_rate", report.hallucination_rate, 0.05, False),
            ("out_of_scope_refusal_rate", report.out_of_scope_refusal_rate, 0.80, True),
        ]

        html = """<table>
<tr>
  <th>Metric</th><th>Value</th><th>Threshold</th><th>Bar</th><th>Status</th>
</tr>"""
        for name, val, thresh, higher_is_better in rows:
            passed = (val >= thresh) if higher_is_better else (val <= thresh)
            status_cls = "pass" if passed else "fail"
            status_txt = "✅ Pass" if passed else "❌ Fail"
            bar_pct = int(min(val, 1.0) * 100)
            bar_color = "#27ae60" if passed else "#e74c3c"
            dir_sym = "≥" if higher_is_better else "≤"
            html += f"""
<tr>
  <td><code>{name}</code></td>
  <td><strong>{val:.4f}</strong></td>
  <td>{dir_sym} {thresh}</td>
  <td>
    <div class="bar-bg">
      <div class="bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
    </div>
  </td>
  <td><span class="badge {status_cls}">{status_txt}</span></td>
</tr>"""
        html += "</table>"
        return html

    def _failed_cases_table(self, report: EvaluationReport) -> str:
        failed = [r for r in report.results if not r.passed_gate]
        if not failed:
            return '<p style="color:#27ae60;padding:8px 0;">✅ All test cases passed!</p>'

        html = """<table>
<tr>
  <th>ID</th><th>Category</th><th>Question</th>
  <th>Faithfulness</th><th>Factuality</th><th>Hallucination</th>
</tr>"""
        for r in failed[:20]:  # cap at 20 rows
            hall_badge = '<span class="badge fail">YES</span>' if r.hallucination_detected else "No"
            html += f"""
<tr>
  <td><code>{r.test_case_id}</code></td>
  <td><span class="badge warn-b">{r.category}</span></td>
  <td title="{r.answer[:200]}">{r.question[:60]}…</td>
  <td>{r.faithfulness:.3f}</td>
  <td>{r.factuality:.3f}</td>
  <td>{hall_badge}</td>
</tr>"""
        html += "</table>"
        return html

    def _drift_section(self, drift: DriftMetrics) -> str:
        alert_color = {"none": "#27ae60", "warning": "#f39c12", "critical": "#e74c3c"}
        color = alert_color.get(drift.alert_level, "#95a5a6")
        return f"""
  <div class="section">
    <h2>📡 Drift Monitor</h2>
    <table>
      <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
      <tr><td>Embedding PSI</td><td>{drift.embedding_psi:.4f}</td>
          <td style="color:{color};">{drift.alert_level.upper()}</td></tr>
      <tr><td>Quality Score KL Divergence</td><td>{drift.response_quality_drift:.4f}</td>
          <td></td></tr>
      <tr><td>Vocabulary Overlap</td><td>{drift.vocab_overlap:.4f}</td><td></td></tr>
      <tr><td>Avg Response Length Δ</td><td>{drift.avg_response_length_delta:+.1f} tokens</td>
          <td></td></tr>
    </table>
  </div>"""

    def _hall_section(self, hall: HallucinationSnapshot) -> str:
        rows = "\n".join(
            f"<tr><td>{cat}</td><td>{rate:.4f}</td>"
            f"<td>{'⚠️' if rate > 0.05 else '✅'}</td></tr>"
            for cat, rate in hall.categories.items()
        )
        return f"""
  <div class="section">
    <h2>🧪 Hallucination Tracker — {hall.run_id}</h2>
    <p>Rate: <strong>{hall.hallucination_rate:.4f}</strong> ({hall.hallucinated_count}/{hall.total_responses}) |
       Trend Δ: {hall.trend_delta:+.4f}</p>
    <br>
    <table>
      <tr><th>Category</th><th>Rate</th><th>Status</th></tr>
      {rows}
    </table>
  </div>"""

    def _cost_section(self, cost: CostSnapshot) -> str:
        return f"""
  <div class="section">
    <h2>💰 Cost Tracker — {cost.run_id}</h2>
    <table>
      <tr><td>Model</td><td><code>{cost.model}</code></td></tr>
      <tr><td>Queries</td><td>{cost.n_queries}</td></tr>
      <tr><td>Input Tokens</td><td>{cost.total_input_tokens:,}</td></tr>
      <tr><td>Output Tokens</td><td>{cost.total_output_tokens:,}</td></tr>
      <tr><td>Estimated Cost</td><td><strong>${cost.estimated_cost_usd:.6f}</strong></td></tr>
      <tr><td>Cost Δ vs prev run</td>
          <td style="color:{'#e74c3c' if cost.cost_delta_pct > 50 else '#27ae60'}">
            {cost.cost_delta_pct:+.1f}%</td></tr>
    </table>
  </div>"""
