"""
evaluation/ package — RAGAS-style RAG evaluation framework.

JD Requirements covered:
  - ML model validation and evaluation frameworks
  - LLM testing: prompt evaluation, response validation
  - RAGAS or similar evaluation metrics
  - Quality gates that can fail a CI/CD build
"""
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.test_cases import HR_TEST_CASES
from evaluation.quality_gates import QualityGate
from evaluation.benchmark_reporter import BenchmarkReporter

__all__ = ["RAGASEvaluator", "HR_TEST_CASES", "QualityGate", "BenchmarkReporter"]
