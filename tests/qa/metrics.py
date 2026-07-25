"""Benchmark metrics and quality-gate helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    shared = 0
    ref_counts: Dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            shared += 1
            ref_counts[token] -= 1
    precision = shared / len(pred_tokens)
    recall = shared / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / float(k)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return sum(1 for doc_id in relevant if doc_id in top_k) / float(len(relevant))


def mean_reciprocal_rank(ranks: Iterable[int]) -> float:
    values = [1.0 / rank for rank in ranks if rank > 0]
    return mean(values) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[index])


def latency_percentiles(values: Sequence[float]) -> Dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": float(mean(values)) if values else 0.0,
    }


def token_usage_summary(prompt_tokens: Sequence[int], completion_tokens: Sequence[int], cost_per_1k_tokens: float = 0.0) -> Dict[str, float]:
    total_prompt = sum(prompt_tokens)
    total_completion = sum(completion_tokens)
    total_tokens = total_prompt + total_completion
    estimated_cost = (total_tokens / 1000.0) * cost_per_1k_tokens
    return {
        "prompt_tokens": float(total_prompt),
        "completion_tokens": float(total_completion),
        "total_tokens": float(total_tokens),
        "estimated_cost": float(estimated_cost),
    }


@dataclass
class QualityGateThresholds:
    min_em: float = 0.85
    min_f1: float = 0.85
    min_precision_at_k: float = 0.80
    min_recall_at_k: float = 0.80
    min_mrr: float = 0.80
    max_latency_p95_ms: float = 100.0
    max_latency_p99_ms: float = 200.0
    max_error_rate: float = 0.02
    max_hallucination_rate: float = 0.02
    max_refusal_rate: float = 0.05


@dataclass
class EvaluationRecord:
    stage: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_reason: str | None = None


def check_quality_gate(results: Mapping[str, float], thresholds: QualityGateThresholds) -> Dict[str, Any]:
    failures: List[str] = []

    if results.get("em", 1.0) < thresholds.min_em:
        failures.append("exact_match")
    if results.get("f1", 1.0) < thresholds.min_f1:
        failures.append("f1")
    if results.get("precision_at_k", 1.0) < thresholds.min_precision_at_k:
        failures.append("precision_at_k")
    if results.get("recall_at_k", 1.0) < thresholds.min_recall_at_k:
        failures.append("recall_at_k")
    if results.get("mrr", 1.0) < thresholds.min_mrr:
        failures.append("mrr")
    if results.get("latency_p95_ms", 0.0) > thresholds.max_latency_p95_ms:
        failures.append("latency_p95_ms")
    if results.get("latency_p99_ms", 0.0) > thresholds.max_latency_p99_ms:
        failures.append("latency_p99_ms")
    if results.get("error_rate", 0.0) > thresholds.max_error_rate:
        failures.append("error_rate")
    if results.get("hallucination_rate", 0.0) > thresholds.max_hallucination_rate:
        failures.append("hallucination_rate")
    if results.get("refusal_rate", 0.0) > thresholds.max_refusal_rate:
        failures.append("refusal_rate")

    return {
        "passed": not failures,
        "failures": failures,
        "results": dict(results),
        "thresholds": thresholds.__dict__,
    }

