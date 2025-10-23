"""
Prompt Comparison Tests: v1 vs v2 Anti-Hallucination Analysis
A/B test to measure effectiveness of anti-hallucination guardrails
"""

import pytest
import sys
import os
import json
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import ModelLoader
from prompt.prompt_library import (
    PROMPT_VERSIONS,
    HALLUCINATION_TEST_CASES,
    rag_generation_prompt,
    hallucination_detection_prompt
)
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def llm():
    """Load real LLM for prompt testing"""
    try:
        loader = ModelLoader()
        return loader.load_llm()
    except Exception as e:
        pytest.skip(f"Could not load LLM: {str(e)}")


@pytest.fixture
def hallucination_evaluator(llm):
    """Create hallucination evaluator using LLM"""
    def evaluate(context: str, answer: str, question: str) -> Dict:
        """
        Evaluate if an answer is grounded in context.
        Returns hallucination score and details.
        """
        prompt = hallucination_detection_prompt.format(
            context=context,
            answer=answer,
            question=question
        )

        try:
            response = llm.invoke(prompt)
            # Try to parse JSON response
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            # Fallback: manual parsing
            content = response.content.lower()
            is_grounded = "true" in content or "grounded" in content
            return {
                "is_grounded": is_grounded,
                "hallucination_score": 0.0 if is_grounded else 1.0,
                "explanation": response.content
            }

    return evaluate


# ============================================================================
# TEST DATA
# ============================================================================

PROMPT_TEST_CASES = [
    {
        "name": "Missing Information Test",
        "question": "What is the company's revenue?",
        "context": "TechCorp was founded in 2010 by Jane Smith. The company has 500 employees.",
        "expected_behavior": "Should NOT hallucinate revenue figures",
        "acceptable_answers": [
            "not available",
            "not mentioned",
            "not specified",
            "not provided",
            "does not contain revenue"
        ],
        "hallucination_keywords": ["million", "$", "billion", "USD", "revenue is"]
    },
    {
        "name": "Direct Answer Test",
        "question": "Who founded the company?",
        "context": "TechCorp was founded in 2010 by Jane Smith.",
        "expected_behavior": "Should extract exact information",
        "acceptable_answers": ["jane smith"],
        "hallucination_keywords": ["john", "bob", "michael", "co-founder"]
    },
    {
        "name": "Partial Information Test",
        "question": "What products does the company sell?",
        "context": "The company develops software. Their main product is an AI platform.",
        "expected_behavior": "Should only mention what's explicitly stated",
        "acceptable_answers": ["ai platform", "software"],
        "hallucination_keywords": ["SaaS", "cloud", "mobile app", "enterprise"]
    },
    {
        "name": "Date Extraction Test",
        "question": "When was the company founded?",
        "context": "The organization started operations in 2010.",
        "expected_behavior": "Should extract date accurately",
        "acceptable_answers": ["2010"],
        "hallucination_keywords": ["2009", "2011", "2012", "early 2000s"]
    },
    {
        "name": "Empty Context Test",
        "question": "What is the CEO's name?",
        "context": "",
        "expected_behavior": "Should handle empty context gracefully",
        "acceptable_answers": ["not available", "no information", "not provided"],
        "hallucination_keywords": ["ceo is", "named", "john", "jane"]
    }
]


# ============================================================================
# PROMPT COMPARISON TESTS
# ============================================================================

@pytest.mark.integration
class TestPromptVersionComparison:
    """Compare different prompt versions for hallucination resistance"""

    def test_v1_vs_v2_hallucination_rate(self, llm, hallucination_evaluator):
        """
        Compare hallucination rates between v1 (original) and v2 (enhanced)
        """
        results = {
            "v1_original": {"hallucinations": 0, "total": 0, "details": []},
            "v2_enhanced": {"hallucinations": 0, "total": 0, "details": []}
        }

        for test_case in PROMPT_TEST_CASES:
            for version_key, version_info in PROMPT_VERSIONS.items():
                # Generate answer using this prompt version
                if version_key == "v1_original":
                    # Simple prompt without guardrails
                    prompt = f"""
                    Answer the following question based on the context:

                    Context: {test_case['context']}
                    Question: {test_case['question']}

                    Answer:
                    """
                else:  # v2_enhanced
                    # Use RAG generation prompt with guardrails
                    prompt = rag_generation_prompt.format(
                        context=test_case['context'],
                        question=test_case['question']
                    )

                # Get LLM response
                response = llm.invoke(prompt)
                answer = response.content.lower()

                # Evaluate hallucination
                evaluation = hallucination_evaluator(
                    test_case['context'],
                    answer,
                    test_case['question']
                )

                # Check for hallucination keywords
                has_hallucination_keywords = any(
                    keyword in answer
                    for keyword in test_case.get('hallucination_keywords', [])
                )

                # Check if answer is acceptable
                is_acceptable = any(
                    acceptable in answer
                    for acceptable in test_case['acceptable_answers']
                )

                # Determine if this is a hallucination
                is_hallucination = (
                    has_hallucination_keywords or
                    (not is_acceptable and test_case['context'] == "") or
                    evaluation.get('hallucination_score', 0) > 0.5
                )

                # Record results
                results[version_key]["total"] += 1
                if is_hallucination:
                    results[version_key]["hallucinations"] += 1

                results[version_key]["details"].append({
                    "test_case": test_case['name'],
                    "question": test_case['question'],
                    "answer": answer[:200],  # First 200 chars
                    "is_hallucination": is_hallucination,
                    "evaluation": evaluation
                })

        # Calculate hallucination rates
        v1_rate = results["v1_original"]["hallucinations"] / results["v1_original"]["total"]
        v2_rate = results["v2_enhanced"]["hallucinations"] / results["v2_enhanced"]["total"]

        print("\n=== Prompt Comparison Results ===")
        print(f"V1 Original Hallucination Rate: {v1_rate:.1%}")
        print(f"V2 Enhanced Hallucination Rate: {v2_rate:.1%}")
        print(f"Improvement: {((v1_rate - v2_rate) / v1_rate * 100):.1f}% reduction")

        # Assert v2 is better than v1
        assert v2_rate < v1_rate, \
            f"V2 should have lower hallucination rate than V1 (v1={v1_rate:.2%}, v2={v2_rate:.2%})"

        # Target: v2 should have <5% hallucination rate
        assert v2_rate < 0.05, \
            f"V2 hallucination rate should be <5%, got {v2_rate:.1%}"

    def test_missing_information_handling(self, llm):
        """Test how each prompt version handles missing information"""
        test_case = {
            "question": "What is the company's revenue in 2023?",
            "context": "TechCorp was founded in 2010. It has offices in 5 cities."
        }

        # V1 - Simple prompt
        v1_prompt = f"Context: {test_case['context']}\n\nQuestion: {test_case['question']}\n\nAnswer:"
        v1_response = llm.invoke(v1_prompt).content

        # V2 - Enhanced prompt with guardrails
        v2_prompt = rag_generation_prompt.format(
            context=test_case['context'],
            question=test_case['question']
        )
        v2_response = llm.invoke(v2_prompt).content

        print("\n=== Missing Information Handling ===")
        print(f"V1 Response: {v1_response}")
        print(f"V2 Response: {v2_response}")

        # V2 should explicitly say information is not available
        v2_lower = v2_response.lower()
        assert any(phrase in v2_lower for phrase in [
            "not available",
            "not mentioned",
            "not provided",
            "does not contain",
            "information is not"
        ]), "V2 should explicitly state when information is not available"

    def test_confidence_calibration(self, llm):
        """Test if prompt versions produce well-calibrated confidence"""
        confident_case = {
            "question": "Who is the CEO?",
            "context": "Alice Johnson is the CEO of TechCorp."
        }

        uncertain_case = {
            "question": "Who is the CFO?",
            "context": "Alice Johnson is the CEO of TechCorp."
        }

        # Test V2 prompt on both cases
        v2_confident = llm.invoke(rag_generation_prompt.format(
            context=confident_case['context'],
            question=confident_case['question']
        )).content

        v2_uncertain = llm.invoke(rag_generation_prompt.format(
            context=uncertain_case['context'],
            question=uncertain_case['question']
        )).content

        print("\n=== Confidence Calibration ===")
        print(f"Confident case response: {v2_confident}")
        print(f"Uncertain case response: {v2_uncertain}")

        # Should be confident when answer is clear
        assert "alice johnson" in v2_confident.lower()

        # Should express uncertainty when answer is not available
        v2_uncertain_lower = v2_uncertain.lower()
        assert any(phrase in v2_uncertain_lower for phrase in [
            "not available",
            "not mentioned",
            "not provided",
            "does not contain"
        ]), "Should express uncertainty when information is missing"


# ============================================================================
# HALLUCINATION TEST CASES FROM LIBRARY
# ============================================================================

@pytest.mark.integration
class TestHallucinationTestCases:
    """Run predefined hallucination test cases against current prompts"""

    @pytest.mark.parametrize("test_case", HALLUCINATION_TEST_CASES)
    def test_hallucination_case(self, llm, test_case):
        """Test each hallucination case from prompt library"""
        # Use V2 enhanced prompt
        prompt = rag_generation_prompt.format(
            context=test_case['context'],
            question=test_case['question']
        )

        response = llm.invoke(prompt).content.lower()

        print(f"\n=== Test: {test_case['name']} ===")
        print(f"Question: {test_case['question']}")
        print(f"Context: {test_case['context']}")
        print(f"Response: {response}")

        # Check expected answer
        if isinstance(test_case['expected_answer'], str):
            assert test_case['expected_answer'].lower() in response, \
                f"Expected '{test_case['expected_answer']}' in response"

        # Check should NOT contain
        if test_case.get('should_NOT_contain'):
            for forbidden in test_case['should_NOT_contain']:
                assert forbidden.lower() not in response, \
                    f"Response should NOT contain '{forbidden}'"


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@pytest.mark.integration
class TestPromptPerformanceMetrics:
    """Measure performance metrics for different prompts"""

    def test_response_quality_metrics(self, llm):
        """Measure response quality across multiple test cases"""
        metrics = {
            "total_tests": 0,
            "correct_answers": 0,
            "false_positives": 0,  # Hallucinated when should say "not available"
            "false_negatives": 0,  # Said "not available" when answer exists
            "well_calibrated": 0   # Appropriate confidence level
        }

        for test_case in PROMPT_TEST_CASES:
            prompt = rag_generation_prompt.format(
                context=test_case['context'],
                question=test_case['question']
            )

            response = llm.invoke(prompt).content.lower()
            metrics["total_tests"] += 1

            # Check if response is acceptable
            is_acceptable = any(
                acceptable in response
                for acceptable in test_case['acceptable_answers']
            )

            # Check for hallucination keywords
            has_hallucination = any(
                keyword in response
                for keyword in test_case.get('hallucination_keywords', [])
            )

            if is_acceptable and not has_hallucination:
                metrics["correct_answers"] += 1
                metrics["well_calibrated"] += 1
            elif has_hallucination:
                metrics["false_positives"] += 1

        # Calculate rates
        accuracy = metrics["correct_answers"] / metrics["total_tests"]
        false_positive_rate = metrics["false_positives"] / metrics["total_tests"]

        print("\n=== Performance Metrics ===")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"False Positive Rate (Hallucinations): {false_positive_rate:.1%}")
        print(f"Well-Calibrated Responses: {metrics['well_calibrated']}/{metrics['total_tests']}")

        # Assertions
        assert accuracy > 0.7, f"Accuracy should be >70%, got {accuracy:.1%}"
        assert false_positive_rate < 0.2, f"False positive rate should be <20%, got {false_positive_rate:.1%}"


# ============================================================================
# A/B TESTING UTILITIES
# ============================================================================

class PromptABTester:
    """Utility for running A/B tests on prompts"""

    def __init__(self, llm):
        self.llm = llm
        self.results = {}

    def run_ab_test(self, test_cases: List[Dict], prompt_a, prompt_b, name_a="A", name_b="B"):
        """
        Run A/B test comparing two prompts.

        Returns:
            Dict with comparison metrics
        """
        results = {
            name_a: {"hallucinations": 0, "total": 0},
            name_b: {"hallucinations": 0, "total": 0}
        }

        for test_case in test_cases:
            # Test prompt A
            response_a = self.llm.invoke(prompt_a.format(
                context=test_case['context'],
                question=test_case['question']
            )).content.lower()

            # Test prompt B
            response_b = self.llm.invoke(prompt_b.format(
                context=test_case['context'],
                question=test_case['question']
            )).content.lower()

            # Evaluate both
            for response, key in [(response_a, name_a), (response_b, name_b)]:
                results[key]["total"] += 1

                # Check for hallucinations
                has_hallucination = any(
                    keyword in response
                    for keyword in test_case.get('hallucination_keywords', [])
                )

                if has_hallucination:
                    results[key]["hallucinations"] += 1

        # Calculate rates
        rate_a = results[name_a]["hallucinations"] / results[name_a]["total"]
        rate_b = results[name_b]["hallucinations"] / results[name_b]["total"]

        return {
            "prompt_a": name_a,
            "prompt_b": name_b,
            "hallucination_rate_a": rate_a,
            "hallucination_rate_b": rate_b,
            "improvement": ((rate_a - rate_b) / rate_a * 100) if rate_a > 0 else 0,
            "winner": name_b if rate_b < rate_a else name_a
        }


@pytest.mark.integration
def test_ab_testing_framework(llm):
    """Test the A/B testing framework"""
    tester = PromptABTester(llm)

    # Create simple test prompts
    prompt_a = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    prompt_b = rag_generation_prompt

    results = tester.run_ab_test(
        PROMPT_TEST_CASES[:3],  # Use first 3 test cases
        prompt_a,
        prompt_b,
        name_a="Simple",
        name_b="Enhanced"
    )

    print("\n=== A/B Test Results ===")
    print(f"Winner: {results['winner']}")
    print(f"Improvement: {results['improvement']:.1f}%")

    assert results['hallucination_rate_b'] <= results['hallucination_rate_a'], \
        "Enhanced prompt should perform at least as well as simple prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
