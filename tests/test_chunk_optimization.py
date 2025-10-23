"""
Chunk Size Optimization Tests
Empirically determine optimal chunk sizes for different document types
"""

import pytest
import sys
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import ModelLoader
from src.document_chat.retrieval_layer import (
    GoogleGenAIEmbeddingProvider,
    InMemoryVectorStore,
    DocumentRetriever,
    SlidingWindowChunker,
    RetrievalMetrics
)


# ============================================================================
# TEST DATA - Different Document Types
# ============================================================================

TEST_DOCUMENTS = {
    "short_technical": {
        "content": """
        TechCorp API Documentation:
        Authentication: Use Bearer token in header.
        Endpoint: POST /api/v1/users
        Request body: {"name": "string", "email": "string"}
        Response: 201 Created with user object.
        Rate limit: 100 requests per minute.
        """,
        "optimal_chunk_range": (150, 300),
        "queries": [
            "How do I authenticate?",
            "What's the rate limit?",
            "What's the user endpoint?"
        ]
    },
    "medium_narrative": {
        "content": """
        TechCorp was founded in 2010 by Alice Johnson and Michael Chen in San Francisco.
        The company initially focused on mobile app development but pivoted to enterprise
        software in 2015 after securing $50 million in Series A funding from Acme Ventures.

        The flagship product, AI Assistant Pro, was launched in 2018 and quickly gained
        traction in the Fortune 500 market. By 2020, TechCorp had expanded to 500 employees
        across offices in San Francisco, New York, London, Tokyo, and Bangalore.

        The company's mission is to democratize AI technology and make it accessible to
        businesses of all sizes. Their core values include innovation, customer-centricity,
        transparency, and continuous learning.
        """,
        "optimal_chunk_range": (300, 600),
        "queries": [
            "When was TechCorp founded?",
            "What is the company mission?",
            "Where are the offices located?"
        ]
    },
    "long_structured": {
        "content": """
        QUARTERLY FINANCIAL REPORT - Q3 2023

        EXECUTIVE SUMMARY
        TechCorp showed strong performance in Q3 2023 with revenue growth of 25% YoY.
        The company achieved profitability for the first time, with a net profit margin of 8%.

        FINANCIAL HIGHLIGHTS
        Revenue: $10 million (up from $8M in Q3 2022)
        Operating Expenses: $7 million
        Net Profit: $800,000
        EBITDA: $1.5 million
        Cash Balance: $25 million

        REVENUE BREAKDOWN BY SEGMENT
        Enterprise Sales: $6 million (60%)
        SMB Sales: $2.5 million (25%)
        Professional Services: $1 million (10%)
        Other: $500,000 (5%)

        OPERATING EXPENSES BREAKDOWN
        Sales & Marketing: $3 million
        Research & Development: $2.5 million
        General & Administrative: $1 million
        Other: $500,000

        OUTLOOK FOR Q4 2023
        The company expects continued growth with revenue target of $12 million.
        Plans to hire 50 additional engineers to support product development.
        Expansion into European markets scheduled for early 2024.
        """,
        "optimal_chunk_range": (400, 800),
        "queries": [
            "What was the Q3 2023 revenue?",
            "What's the revenue breakdown?",
            "What are the Q4 plans?"
        ]
    }
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def embedding_provider():
    """Load real embedding provider"""
    try:
        loader = ModelLoader()
        embedder = loader.load_embeddings()
        return GoogleGenAIEmbeddingProvider(embedder)
    except Exception as e:
        pytest.skip(f"Could not load embeddings: {str(e)}")


# ============================================================================
# CHUNK SIZE EVALUATION
# ============================================================================

def evaluate_chunk_size(
    doc_content: str,
    doc_id: str,
    queries: List[str],
    chunk_size: int,
    overlap: int,
    embedding_provider
) -> Dict[str, float]:
    """
    Evaluate retrieval quality for a specific chunk size.

    Returns:
        Dict with quality metrics
    """
    # Create retriever with specific chunk size
    chunker = SlidingWindowChunker(chunk_size=chunk_size, overlap=overlap, doc_id=doc_id)
    vector_store = InMemoryVectorStore()
    retriever = DocumentRetriever(embedding_provider, vector_store, chunker)

    # Index document
    num_chunks = retriever.index_document(doc_id, doc_content, {})

    # Test all queries
    all_similarities = []
    all_ranks = []

    for query in queries:
        results = retriever.retrieve(query, top_k=3)

        if results:
            all_similarities.append(results[0].similarity_score)
            all_ranks.append(1)  # Simplified - assuming first result is relevant

    # Calculate metrics
    avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
    mrr = RetrievalMetrics.mean_reciprocal_rank(all_ranks) if all_ranks else 0.0

    return {
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "avg_similarity": avg_similarity,
        "mrr": mrr,
        "queries_tested": len(queries)
    }


# ============================================================================
# OPTIMIZATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.parametrize("chunk_size", [128, 256, 384, 512, 768, 1024])
class TestChunkSizeOptimization:
    """Test different chunk sizes to find optimal configuration"""

    def test_chunk_size_impact_on_retrieval(self, embedding_provider, chunk_size):
        """Test how chunk size affects retrieval quality"""
        doc_type = "medium_narrative"
        doc_info = TEST_DOCUMENTS[doc_type]

        metrics = evaluate_chunk_size(
            doc_content=doc_info["content"],
            doc_id=f"test_{doc_type}",
            queries=doc_info["queries"],
            chunk_size=chunk_size,
            overlap=50,
            embedding_provider=embedding_provider
        )

        print(f"\n=== Chunk Size: {chunk_size} ===")
        print(f"Num Chunks: {metrics['num_chunks']}")
        print(f"Avg Similarity: {metrics['avg_similarity']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")

        # Basic sanity checks
        assert metrics['num_chunks'] > 0, "Should create at least one chunk"
        assert metrics['avg_similarity'] > 0, "Should have positive similarity"


@pytest.mark.integration
class TestComprehensiveChunkOptimization:
    """Comprehensive chunk size optimization across document types"""

    def test_find_optimal_chunk_size_per_document_type(self, embedding_provider):
        """
        Find optimal chunk size for each document type.
        This test runs multiple configurations and identifies the best.
        """
        chunk_sizes = [128, 256, 384, 512, 768, 1024]
        results_by_doc_type = {}

        for doc_type, doc_info in TEST_DOCUMENTS.items():
            print(f"\n{'='*60}")
            print(f"Testing Document Type: {doc_type}")
            print(f"{'='*60}")

            results = []

            for chunk_size in chunk_sizes:
                metrics = evaluate_chunk_size(
                    doc_content=doc_info["content"],
                    doc_id=f"test_{doc_type}",
                    queries=doc_info["queries"],
                    chunk_size=chunk_size,
                    overlap=50,
                    embedding_provider=embedding_provider
                )

                results.append(metrics)

                print(f"Chunk Size: {chunk_size:4d} | "
                      f"Chunks: {metrics['num_chunks']:2d} | "
                      f"Similarity: {metrics['avg_similarity']:.3f} | "
                      f"MRR: {metrics['mrr']:.3f}")

            # Find best chunk size (highest avg similarity)
            best_result = max(results, key=lambda x: x['avg_similarity'])

            print(f"\n✅ Best chunk size for {doc_type}: {best_result['chunk_size']}")
            print(f"   Similarity: {best_result['avg_similarity']:.3f}")
            print(f"   Optimal range: {doc_info['optimal_chunk_range']}")

            results_by_doc_type[doc_type] = {
                "best_chunk_size": best_result['chunk_size'],
                "best_similarity": best_result['avg_similarity'],
                "expected_range": doc_info['optimal_chunk_range'],
                "all_results": results
            }

            # Verify best chunk size is in expected range
            optimal_min, optimal_max = doc_info['optimal_chunk_range']
            assert optimal_min <= best_result['chunk_size'] <= optimal_max, \
                f"Best chunk size {best_result['chunk_size']} outside expected range {doc_info['optimal_chunk_range']}"

        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)

        for doc_type, result in results_by_doc_type.items():
            print(f"{doc_type:20s} | Best: {result['best_chunk_size']:4d} | "
                  f"Similarity: {result['best_similarity']:.3f}")

    def test_overlap_impact(self, embedding_provider):
        """Test how overlap affects retrieval quality"""
        doc_type = "medium_narrative"
        doc_info = TEST_DOCUMENTS[doc_type]

        chunk_size = 512
        overlaps = [0, 25, 50, 100, 150]

        print(f"\n{'='*60}")
        print(f"Testing Overlap Impact (Chunk Size: {chunk_size})")
        print(f"{'='*60}")

        results = []

        for overlap in overlaps:
            chunker = SlidingWindowChunker(
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id="test_overlap"
            )
            vector_store = InMemoryVectorStore()
            retriever = DocumentRetriever(embedding_provider, vector_store, chunker)

            num_chunks = retriever.index_document("test_overlap", doc_info["content"], {})

            similarities = []
            for query in doc_info["queries"]:
                retrieval_results = retriever.retrieve(query, top_k=3)
                if retrieval_results:
                    similarities.append(retrieval_results[0].similarity_score)

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            print(f"Overlap: {overlap:3d} | Chunks: {num_chunks:2d} | "
                  f"Similarity: {avg_similarity:.3f}")

            results.append({
                "overlap": overlap,
                "num_chunks": num_chunks,
                "avg_similarity": avg_similarity
            })

        # Best overlap should be in 25-100 range
        best_overlap = max(results, key=lambda x: x['avg_similarity'])['overlap']
        print(f"\n✅ Best overlap: {best_overlap}")

        assert 25 <= best_overlap <= 100, \
            f"Best overlap {best_overlap} should be in range 25-100"

    def test_chunk_size_tradeoff_analysis(self, embedding_provider):
        """
        Analyze tradeoffs between chunk size, number of chunks, and quality.
        """
        doc_type = "long_structured"
        doc_info = TEST_DOCUMENTS[doc_type]

        chunk_sizes = [128, 256, 384, 512, 768, 1024, 1536]
        overlap = 50

        metrics_data = []

        print(f"\n{'='*70}")
        print(f"Chunk Size Tradeoff Analysis")
        print(f"{'='*70}")
        print(f"{'Size':>6} | {'Chunks':>6} | {'Similarity':>10} | {'Storage':>10} | {'Quality/Chunk':>12}")
        print("-" * 70)

        for chunk_size in chunk_sizes:
            metrics = evaluate_chunk_size(
                doc_content=doc_info["content"],
                doc_id=f"test_tradeoff",
                queries=doc_info["queries"],
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_provider=embedding_provider
            )

            # Calculate efficiency metric (quality per chunk)
            efficiency = metrics['avg_similarity'] / metrics['num_chunks'] if metrics['num_chunks'] > 0 else 0

            metrics_data.append({
                "chunk_size": chunk_size,
                "num_chunks": metrics['num_chunks'],
                "similarity": metrics['avg_similarity'],
                "efficiency": efficiency
            })

            print(f"{chunk_size:6d} | {metrics['num_chunks']:6d} | "
                  f"{metrics['avg_similarity']:10.3f} | "
                  f"{metrics['num_chunks'] * chunk_size:10d} | "
                  f"{efficiency:12.4f}")

        # Find best tradeoff (highest efficiency)
        best_tradeoff = max(metrics_data, key=lambda x: x['efficiency'])

        print(f"\n✅ Best tradeoff chunk size: {best_tradeoff['chunk_size']}")
        print(f"   Efficiency: {best_tradeoff['efficiency']:.4f}")
        print(f"   Similarity: {best_tradeoff['similarity']:.3f}")
        print(f"   Num Chunks: {best_tradeoff['num_chunks']}")


# ============================================================================
# VISUALIZATION (Optional)
# ============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Visualization test - run manually")
def test_visualize_chunk_size_impact(embedding_provider):
    """
    Create visualization of chunk size impact.
    Run with: pytest tests/test_chunk_optimization.py::test_visualize_chunk_size_impact -v -s
    """
    doc_type = "medium_narrative"
    doc_info = TEST_DOCUMENTS[doc_type]

    chunk_sizes = range(128, 1537, 128)
    similarities = []
    num_chunks_list = []

    for chunk_size in chunk_sizes:
        metrics = evaluate_chunk_size(
            doc_content=doc_info["content"],
            doc_id=f"test_viz",
            queries=doc_info["queries"],
            chunk_size=chunk_size,
            overlap=50,
            embedding_provider=embedding_provider
        )

        similarities.append(metrics['avg_similarity'])
        num_chunks_list.append(metrics['num_chunks'])

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Chunk size vs similarity
    ax1.plot(chunk_sizes, similarities, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Chunk Size (characters)', fontsize=12)
    ax1.set_ylabel('Average Similarity Score', fontsize=12)
    ax1.set_title('Chunk Size vs Retrieval Quality', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=512, color='r', linestyle='--', alpha=0.5, label='Default (512)')
    ax1.legend()

    # Plot 2: Chunk size vs number of chunks
    ax2.plot(chunk_sizes, num_chunks_list, marker='s', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Chunk Size (characters)', fontsize=12)
    ax2.set_ylabel('Number of Chunks', fontsize=12)
    ax2.set_title('Chunk Size vs Storage Overhead', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=512, color='r', linestyle='--', alpha=0.5, label='Default (512)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('chunk_size_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved to: chunk_size_analysis.png")


# ============================================================================
# RECOMMENDATIONS
# ============================================================================

@pytest.mark.integration
def test_generate_chunk_size_recommendations(embedding_provider):
    """
    Generate evidence-based recommendations for chunk sizes.
    """
    print("\n" + "="*70)
    print("CHUNK SIZE RECOMMENDATIONS")
    print("="*70)

    recommendations = {
        "short_technical": {
            "recommended_size": 256,
            "overlap": 50,
            "rationale": "Short, dense technical content benefits from smaller chunks"
        },
        "medium_narrative": {
            "recommended_size": 512,
            "overlap": 50,
            "rationale": "Balanced size for narrative content with context preservation"
        },
        "long_structured": {
            "recommended_size": 768,
            "overlap": 100,
            "rationale": "Larger chunks capture complete sections in structured documents"
        }
    }

    for doc_type, rec in recommendations.items():
        # Verify recommendation
        doc_info = TEST_DOCUMENTS[doc_type]
        metrics = evaluate_chunk_size(
            doc_content=doc_info["content"],
            doc_id=f"test_rec_{doc_type}",
            queries=doc_info["queries"],
            chunk_size=rec["recommended_size"],
            overlap=rec["overlap"],
            embedding_provider=embedding_provider
        )

        print(f"\n{doc_type.upper()}")
        print(f"  Recommended Size: {rec['recommended_size']}")
        print(f"  Overlap: {rec['overlap']}")
        print(f"  Rationale: {rec['rationale']}")
        print(f"  Measured Similarity: {metrics['avg_similarity']:.3f}")
        print(f"  Number of Chunks: {metrics['num_chunks']}")

        # Verify quality is reasonable
        assert metrics['avg_similarity'] > 0.6, \
            f"Recommended chunk size should achieve >0.6 similarity for {doc_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
