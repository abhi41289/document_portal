"""
bias_detection/ package — LLM response bias and fairness testing.

JD Requirements covered:
  - LLM testing: bias detection
  - Prompt fairness checker
  - Response consistency scoring
"""
from bias_detection.bias_detector import BiasDetector
from bias_detection.prompt_fairness import PromptFairnessChecker
from bias_detection.consistency_scorer import ConsistencyScorer

__all__ = ["BiasDetector", "PromptFairnessChecker", "ConsistencyScorer"]
