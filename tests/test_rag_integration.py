"""
Integration Tests for RAG Pipeline with Real Embeddings
Tests the complete retrieval pipeline using actual Google Generative AI embeddings
"""

import pytest
import sys
import os
from typing import List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import ModelLoader
from src.document_chat.retrieval_layer import (
    GoogleGenAIEmbeddingProvider,
    InMemoryVectorStore,
    FAISSVectorStore,
    DocumentRetriever,
    SlidingWindowChunker,
    SemanticChunker,
    RetrievalMetrics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def real_embedder():
    """Load real Google Generative AI embeddings"""
    try:
        loader = ModelLoader()
        embedder = loader.load_embeddings()
        return embedder
    except Exception as e:
        pytest.skip(f"Could not load real embeddings: {str(e)}")


@pytest.fixture
def embedding_provider(real_embedder):
    """Create embedding provider with real embeddings"""
    return GoogleGenAIEmbeddingProvider(real_embedder)


@pytest.fixture
def vector_store():
    """Create in-memory vector store"""
    return InMemoryVectorStore()


@pytest.fixture
def faiss_vector_store():
    """Create FAISS vector store (if available)"""
    try:
        return FAISSVectorStore()
    except ImportError:
        pytest.skip("FAISS not installed")


@pytest.fixture
def retriever(embedding_provider, vector_store):
    """Create document retriever with real components"""
    return DocumentRetriever(
        embedder=embedding_provider,
        vector_store=vector_store,
        chunking_strategy=SlidingWindowChunker(chunk_size=512, overlap=50)
    )


# ============================================================================
# TEST DATA
# ============================================================================

SAMPLE_DOCUMENTS = {
    "doc_001": """
    TechCorp was founded in 2010 by Jane Smith and Michael Chen in Silicon Valley.
    The company specializes in artificial intelligence and machine learning solutions.
    In 2015, TechCorp raised $50 million in Series A funding led by Acme Ventures.
    The company's flagship product, AI Assistant Pro, was launched in 2018.
    As of 2023, TechCorp has over 500 employees across 5 global offices.
    """,

    "doc_002": """
    The quarterly financial report shows strong growth in Q3 2023.
    Revenue increased by 25% year-over-year to $10 million.
    Operating expenses were $7 million, resulting in a profit margin of 30%.
    The company expects to reach profitability by Q4 2024.
    Key growth drivers include enterprise sales and international expansion.
    """,

    "doc_003": """
    Product Roadmap for 2024:
    - Q1: Launch mobile app with offline support
    - Q2: Integrate advanced NLP capabilities
    - Q3: Release API for third-party integrations
    - Q4: Deploy multi-language support for 10+ languages
    Customer feedback indicates high demand for real-time collaboration features.
    """
}


TEST_QUERIES = [
    {
        "query": "When was TechCorp founded?",
        "expected_doc": "doc_001",
        "expected_answer_contains": ["2010", "founded"]
    },
    {
        "query": "What is the revenue for Q3 2023?",
        "expected_doc": "doc_002",
        "expected_answer_contains": ["$10 million", "revenue"]
    },
    {
        "query": "What features are planned for Q2 2024?",
        "expected_doc": "doc_003",
        "expected_answer_contains": ["Q2", "NLP"]
    },
]


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestEndToEndRAGPipeline:
    """Test complete RAG pipeline with real embeddings"""

    def test_index_and_retrieve_single_document(self, retriever):
        """Test indexing a document and retrieving relevant chunks"""
        # Index document
        doc_text = SAMPLE_DOCUMENTS["doc_001"]
        num_chunks = retriever.index_document("doc_001", doc_text, {"source": "test"})

        assert num_chunks > 0, "Should create at least one chunk"

        # Retrieve
        query = "When was TechCorp founded?"
        results = retriever.retrieve(query, top_k=3)

        assert len(results) > 0, "Should return at least one result"
        assert results[0].similarity_score > 0.5, "Top result should have high similarity"
        assert "2010" in results[0].chunk.content, "Should contain founding year"

    def test_multi_document_retrieval(self, retriever):
        """Test retrieval across multiple documents"""
        # Index all documents
        for doc_id, doc_text in SAMPLE_DOCUMENTS.items():
            retriever.index_document(doc_id, doc_text, {"source": "test"})

        # Test each query
        for test_case in TEST_QUERIES:
            results = retriever.retrieve(test_case["query"], top_k=3)

            assert len(results) > 0, f"No results for query: {test_case['query']}"

            # Check if expected document is in top results
            top_doc_ids = [r.chunk.source_document_id for r in results]
            assert test_case["expected_doc"] in top_doc_ids, \
                f"Expected doc {test_case['expected_doc']} not in top results for: {test_case['query']}"

            # Check if expected content is present
            top_content = " ".join([r.chunk.content for r in results[:3]])
            for expected_text in test_case["expected_answer_contains"]:
                assert expected_text.lower() in top_content.lower(), \
                    f"Expected '{expected_text}' in retrieved content for: {test_case['query']}"

    def test_retrieval_quality_metrics(self, retriever):
        """Test retrieval quality using metrics"""
        # Index documents
        for doc_id, doc_text in SAMPLE_DOCUMENTS.items():
            retriever.index_document(doc_id, doc_text, {"source": "test"})

        # Collect similarity scores
        all_similarities = []

        for test_case in TEST_QUERIES:
            results = retriever.retrieve(test_case["query"], top_k=5)
            similarities = [r.similarity_score for r in results]
            all_similarities.extend(similarities)

        # Analyze distribution
        distribution = RetrievalMetrics.similarity_distribution(all_similarities)

        assert distribution["mean"] > 0.4, "Average similarity should be reasonable"
        assert distribution["max"] > 0.6, "Best matches should have high similarity"
        assert distribution["min"] > 0.0, "All similarities should be positive"

    def test_empty_query_handling(self, retriever):
        """Test handling of edge cases"""
        # Index document
        retriever.index_document("doc_001", SAMPLE_DOCUMENTS["doc_001"], {})

        # Empty query should return empty results or handle gracefully
        try:
            results = retriever.retrieve("", top_k=3)
            assert isinstance(results, list), "Should return a list"
        except Exception:
            pass  # Some providers may raise error for empty query

    def test_document_removal(self, retriever):
        """Test removing documents from index"""
        # Index document
        retriever.index_document("doc_001", SAMPLE_DOCUMENTS["doc_001"], {})

        # Verify retrieval works
        results_before = retriever.retrieve("TechCorp", top_k=3)
        assert len(results_before) > 0

        # Remove document
        retriever.remove_document("doc_001")

        # Verify document is gone
        results_after = retriever.retrieve("TechCorp", top_k=3)
        assert len(results_after) == 0


# ============================================================================
# EMBEDDING PROVIDER INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestEmbeddingProviderIntegration:
    """Test embedding provider with real Google embeddings"""

    def test_embed_single_text(self, embedding_provider):
        """Test single text embedding"""
        text = "This is a test document about artificial intelligence."
        embedding = embedding_provider.embed_text(text)

        assert isinstance(embedding, list), "Should return list"
        assert len(embedding) > 0, "Embedding should have dimensions"
        assert all(isinstance(x, float) for x in embedding), "All values should be floats"

    def test_embed_batch(self, embedding_provider):
        """Test batch embedding"""
        texts = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Natural language processing handles text."
        ]

        embeddings = embedding_provider.embed_batch(texts)

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        assert all(len(emb) == len(embeddings[0]) for emb in embeddings), \
            "All embeddings should have same dimension"

    def test_semantic_similarity(self, embedding_provider):
        """Test that semantically similar texts have similar embeddings"""
        text1 = "The company was founded in 2010"
        text2 = "The organization started in two thousand ten"
        text3 = "The weather is sunny today"

        emb1 = embedding_provider.embed_text(text1)
        emb2 = embedding_provider.embed_text(text2)
        emb3 = embedding_provider.embed_text(text3)

        # Calculate cosine similarity
        def cosine_sim(v1, v2):
            import numpy as np
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        sim_1_2 = cosine_sim(emb1, emb2)
        sim_1_3 = cosine_sim(emb1, emb3)

        assert sim_1_2 > sim_1_3, \
            "Similar texts should have higher similarity than unrelated texts"


# ============================================================================
# CHUNKING STRATEGY INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestChunkingStrategies:
    """Test different chunking strategies with real documents"""

    def test_sliding_window_chunking(self, embedding_provider, vector_store):
        """Test sliding window chunking strategy"""
        chunker = SlidingWindowChunker(chunk_size=200, overlap=50, doc_id="doc_001")
        retriever = DocumentRetriever(embedding_provider, vector_store, chunker)

        num_chunks = retriever.index_document("doc_001", SAMPLE_DOCUMENTS["doc_001"], {})

        assert num_chunks > 0, "Should create chunks"

        # Test retrieval
        results = retriever.retrieve("founded in 2010", top_k=3)
        assert len(results) > 0, "Should retrieve results"

    def test_semantic_chunking(self, embedding_provider, vector_store):
        """Test semantic chunking strategy"""
        chunker = SemanticChunker(doc_id="doc_003", target_size=300)
        retriever = DocumentRetriever(embedding_provider, vector_store, chunker)

        num_chunks = retriever.index_document("doc_003", SAMPLE_DOCUMENTS["doc_003"], {})

        assert num_chunks > 0, "Should create semantic chunks"

        # Test retrieval
        results = retriever.retrieve("product roadmap", top_k=3)
        assert len(results) > 0, "Should retrieve results"


# ============================================================================
# VECTOR STORE COMPARISON TESTS
# ============================================================================

@pytest.mark.integration
class TestVectorStoreComparison:
    """Compare InMemory vs FAISS vector stores"""

    def test_inmemory_vs_faiss(self, embedding_provider, faiss_vector_store):
        """Compare results from both vector stores"""
        # Create retrievers with different stores
        inmemory_store = InMemoryVectorStore()
        retriever_inmemory = DocumentRetriever(embedding_provider, inmemory_store)
        retriever_faiss = DocumentRetriever(embedding_provider, faiss_vector_store)

        # Index same document in both
        doc_text = SAMPLE_DOCUMENTS["doc_001"]
        retriever_inmemory.index_document("doc_001", doc_text, {})
        retriever_faiss.index_document("doc_001", doc_text, {})

        # Query both
        query = "When was TechCorp founded?"
        results_inmemory = retriever_inmemory.retrieve(query, top_k=3)
        results_faiss = retriever_faiss.retrieve(query, top_k=3)

        # Both should return results
        assert len(results_inmemory) > 0
        assert len(results_faiss) > 0

        # Top results should be similar (not necessarily identical due to different similarity metrics)
        assert results_inmemory[0].chunk.source_document_id == results_faiss[0].chunk.source_document_id


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.integration
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Benchmark retrieval performance"""

    def test_indexing_performance(self, retriever, benchmark):
        """Benchmark document indexing speed"""
        doc_text = SAMPLE_DOCUMENTS["doc_001"] * 10  # Make it larger

        result = benchmark(retriever.index_document, "doc_perf", doc_text, {})

        assert result > 0, "Should create chunks"

    def test_retrieval_performance(self, retriever, benchmark):
        """Benchmark retrieval speed"""
        # Index documents first
        for doc_id, doc_text in SAMPLE_DOCUMENTS.items():
            retriever.index_document(doc_id, doc_text, {})

        # Benchmark retrieval
        results = benchmark(retriever.retrieve, "TechCorp founded", 5)

        assert len(results) > 0, "Should return results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
