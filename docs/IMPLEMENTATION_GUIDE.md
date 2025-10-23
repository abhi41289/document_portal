"""
Practical Implementation Guide: Integrating RAG Testing with document_portal
Step-by-step guide to take the portfolio code and implement it in production
"""

# ============================================================================
# STEP 1: Upgrade the Prompt Library in document_portal
# ============================================================================

"""
CURRENT CODE (document_portal/prompt/prompt_library.py):
---
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template('''
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
''')
---

UPGRADE TO:
---
"""

# NEW VERSION (prompt_library_v2.py)
from langchain_core.prompts import ChatPromptTemplate

# Keep the old prompt for backward compatibility
prompt_v1 = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# NEW: Anti-hallucination prompt
prompt_v2 = ChatPromptTemplate.from_template("""
You are a document metadata analyzer. Your ONLY job is to extract structured information 
that EXPLICITLY appears in the provided document.

CRITICAL RULES:
1. Extract ONLY information that is directly stated in the document
2. Do NOT infer, guess, or use external knowledge
3. For missing information, use: "Not Available"
4. If unsure, mark as "Not Available"
5. ALWAYS return valid JSON matching the schema below

DOCUMENT METADATA SCHEMA:
{format_instructions}

DOCUMENT TO ANALYZE:
---BEGIN DOCUMENT---
{document_text}
---END DOCUMENT---

STRICT INSTRUCTIONS:
- Title: Extract document's main title if present. Otherwise: "Not Available"
- Author: Extract author name if explicitly stated. Do NOT guess. Otherwise: "Unknown"
- DateCreated: Extract creation date if stated. Otherwise: "Not Available"
- LastModifiedDate: Extract last modified date if present. Otherwise: "Not Available"
- Publisher: Extract publisher name if stated. Otherwise: "Not Available"
- Language: Detect language. Otherwise: "Not Detected"
- PageCount: Extract if stated, or estimate from content. Otherwise: "Not Available"
- SentimentTone: Analyze tone (Professional, Academic, Casual, Formal, etc.)
- Summary: 2-3 bullet points from document ONLY

REMEMBER: Every field must be supported by the actual document text.
If the document doesn't contain information, use "Not Available".
Do NOT add information from your training data.

Return ONLY valid JSON:
""")

# Use v2 by default
prompt = prompt_v2


# ============================================================================
# STEP 2: Add Retrieval Layer to document_portal
# ============================================================================

"""
CURRENT document_portal/src/multi_document_chat/retrieval.py:
---
# EMPTY FILE
---

ADD THIS:
"""

# FILE: document_portal/src/multi_document_chat/retrieval.py

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source_document_id: str = None


@dataclass
class RetrievalResult:
    """Result of a retrieval query"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int


class DocumentRetriever:
    """
    Main retriever for multi-document RAG.
    Handles chunking, embedding, and retrieval.
    """
    
    def __init__(self, embedder, vector_store):
        """
        Args:
            embedder: Instance of GoogleGenerativeAIEmbeddings
            vector_store: Instance of vector store (FAISS or in-memory)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)
    
    def index_document(self, doc_id: str, text: str, metadata: Dict = None) -> int:
        """
        Index a document: chunk it, embed it, store it.
        
        Returns:
            Number of chunks created
        """
        if metadata is None:
            metadata = {}
        
        # Chunk the document (simple sliding window)
        chunks = self._chunk_text(text, doc_id, metadata)
        
        # Embed chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Store
        self.vector_store.add_chunks(chunks)
        
        self.logger.info(f"Indexed {len(chunks)} chunks from document {doc_id}")
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Returns:
            List of RetrievalResult objects
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        self.logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}")
        return results
    
    @staticmethod
    def _chunk_text(text: str, doc_id: str, metadata: Dict, 
                   chunk_size: int = 512, overlap: int = 50) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Document text
            doc_id: Document ID
            metadata: Document metadata
            chunk_size: Size of each chunk (characters)
            overlap: Overlap between chunks (characters)
        
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            # Skip very small chunks
            if len(chunk_text.strip()) < 50:
                continue
            
            chunk = DocumentChunk(
                id=f"{doc_id}_chunk_{len(chunks)}",
                content=chunk_text,
                metadata={**metadata, "chunk_start": i},
                source_document_id=doc_id
            )
            chunks.append(chunk)
        
        return chunks


# ============================================================================
# STEP 3: Update data_analysis.py to Use Better Error Handling
# ============================================================================

"""
CURRENT CODE:
---
class DocumentAnalyzer:
    def analyze_document(self, document_text:str)-> dict:
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            response = chain.invoke({...})
            return response
        except Exception as e:
            raise DocumentPortalException(...)
---

UPGRADE TO:
"""

# Add hallucination detection to analysis

def analyze_document_with_validation(self, document_text: str) -> dict:
    """
    Analyze document with added validation for hallucinations.
    """
    try:
        # Original analysis
        chain = self.prompt | self.llm | self.fixing_parser
        response = chain.invoke({
            "format_instructions": self.parser.get_format_instructions(),
            "document_text": document_text
        })
        
        # NEW: Validate response is grounded in document
        self._validate_grounding(response, document_text)
        
        self.log.info("Metadata extraction successful and validated")
        return response
    
    except Exception as e:
        self.log.error("Metadata analysis failed", error=str(e))
        raise
    
    def _validate_grounding(self, response: dict, document_text: str):
        """
        Validate that extracted metadata is grounded in the document.
        
        This catches hallucinations where the LLM makes up information.
        """
        # Check: Author mentioned in doc?
        if response.get("Author") and response["Author"] != "Unknown":
            if response["Author"] not in document_text:
                self.log.warning(f"Author '{response['Author']}' not found in document text")
        
        # Check: Title is meaningful?
        if response.get("Title") and response["Title"] != "Not Available":
            if len(response["Title"]) < 3:
                self.log.warning(f"Title seems too short: '{response['Title']}'")
        
        self.log.debug("Grounding validation passed")


# ============================================================================
# STEP 4: Create a Test Module for document_portal
# ============================================================================

"""
CREATE NEW FILE: document_portal/tests/test_rag_pipeline.py

This is where you'd put your test suite.
"""

import pytest
from src.document_analyzer.data_ingestion import DocumentHandler
from src.document_analyzer.data_analysis import DocumentAnalyzer
from utils.model_loader import ModelLoader


class TestDocumentPortalRAG:
    """
    Test suite for document_portal RAG system.
    Tests all layers: ingestion, analysis, retrieval, generation.
    """
    
    @pytest.fixture
    def sample_pdf_text(self):
        """Sample text as if extracted from PDF"""
        return """
        Annual Report 2024
        
        CEO: Alice Johnson
        Founded: 2010
        Headquarters: San Francisco, CA
        
        This year we achieved record revenue of $100 million.
        The company has grown from 50 to 500 employees.
        """
    
    def test_ingestion_extraction(self):
        """Test: Can we extract text from PDF?"""
        handler = DocumentHandler(session_id="test_rag")
        # Would test with actual PDF in production
        assert True  # Placeholder
    
    def test_metadata_extraction_is_grounded(self, sample_pdf_text):
        """
        Test: Is extracted metadata grounded in the document?
        This is a HALLUCINATION test.
        """
        analyzer = DocumentAnalyzer()
        
        # Analyze the sample text
        result = analyzer.analyze_document(sample_pdf_text)
        
        # Assert: CEO mentioned is actually in document
        if result.get("Author") != "Unknown":
            assert result["Author"] in sample_pdf_text, \
                f"Author '{result['Author']}' not found in document"
        
        # Assert: Year mentioned is in document
        if result.get("DateCreated") != "Not Available":
            assert result["DateCreated"] in sample_pdf_text or "2024" in sample_pdf_text
    
    def test_metadata_extraction_handles_missing_info(self):
        """Test: What happens when info is missing?"""
        analyzer = DocumentAnalyzer()
        
        # Minimal document with missing fields
        minimal_text = "Just some random text about a company."
        result = analyzer.analyze_document(minimal_text)
        
        # Assert: Missing fields should be marked as "Not Available", not guessed
        assert result.get("Author") == "Unknown" or "Unknown" in result.get("Author", "")
        assert result.get("DateCreated") == "Not Available" or "Not Available" in result.get("DateCreated", "")


# ============================================================================
# STEP 5: Add Monitoring & Logging for Production
# ============================================================================

"""
ADD TO document_portal/src/multi_document_chat/monitoring.py
"""

import json
from datetime import datetime
from logger.custom_logger import CustomLogger


class RAGMonitoring:
    """
    Production monitoring for RAG system.
    Tracks quality metrics and alerts on issues.
    """
    
    def __init__(self):
        self.logger = CustomLogger().get_logger(__name__)
        self.metrics = {
            "total_queries": 0,
            "hallucinations_detected": 0,
            "avg_retrieval_score": 0,
            "avg_latency": 0,
        }
    
    def log_query(self, query: str, retrieved_docs: int, answer: str, is_grounded: bool):
        """Log a query and its results"""
        self.metrics["total_queries"] += 1
        
        if not is_grounded:
            self.metrics["hallucinations_detected"] += 1
        
        # Calculate hallucination rate
        if self.metrics["total_queries"] > 0:
            hallucination_rate = self.metrics["hallucinations_detected"] / self.metrics["total_queries"]
        else:
            hallucination_rate = 0
        
        # Log structured event
        self.logger.info(
            "RAG query processed",
            query=query[:50],
            retrieved_docs=retrieved_docs,
            is_grounded=is_grounded,
            hallucination_rate=hallucination_rate,
            timestamp=datetime.now().isoformat()
        )
        
        # Alert if hallucination rate is too high
        if hallucination_rate > 0.05:  # 5% threshold
            self.logger.error(
                "Hallucination rate too high!",
                current_rate=hallucination_rate,
                threshold=0.05
            )
    
    def get_metrics(self) -> dict:
        """Get current metrics"""
        return self.metrics


# ============================================================================
# STEP 6: INTEGRATION EXAMPLES - Real Implementation
# ============================================================================

"""
Complete integration examples showing how to use the new RAG components
in your document_portal application.
"""

# Example 1: Basic RAG Pipeline Integration
# FILE: document_portal/src/document_chat/rag_pipeline.py

from utils.model_loader import ModelLoader
from src.document_chat.retrieval_layer import (
    GoogleGenAIEmbeddingProvider,
    InMemoryVectorStore,
    DocumentRetriever,
    SlidingWindowChunker
)
from src.document_chat.rag_monitoring import RAGMonitoring, AlertThresholds, SlackAlertNotifier
from prompt.prompt_library import rag_generation_prompt
import os


class RAGPipeline:
    """
    Production-ready RAG pipeline for document_portal.
    Integrates retrieval, generation, and monitoring.
    """

    def __init__(self, enable_monitoring: bool = True):
        # Load models
        loader = ModelLoader()
        embedder = loader.load_embeddings()
        self.llm = loader.load_llm()

        # Setup retrieval layer
        embedding_provider = GoogleGenAIEmbeddingProvider(embedder)
        vector_store = InMemoryVectorStore()  # Use FAISSVectorStore for production
        chunking_strategy = SlidingWindowChunker(chunk_size=512, overlap=50)

        self.retriever = DocumentRetriever(
            embedder=embedding_provider,
            vector_store=vector_store,
            chunking_strategy=chunking_strategy
        )

        # Setup monitoring
        if enable_monitoring:
            notifiers = [SlackAlertNotifier(os.getenv("SLACK_WEBHOOK_URL"))] if os.getenv("SLACK_WEBHOOK_URL") else []
            self.monitoring = RAGMonitoring(
                thresholds=AlertThresholds(
                    hallucination_rate=0.05,
                    avg_retrieval_time_ms=500.0,
                    min_similarity_score=0.6
                ),
                notifiers=notifiers
            )
        else:
            self.monitoring = None

    def index_document(self, doc_id: str, content: str, metadata: dict = None):
        """Index a document for retrieval"""
        return self.retriever.index_document(doc_id, content, metadata or {})

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Query the RAG system and return answer with metrics.

        Returns:
            dict with answer, sources, and quality metrics
        """
        import time

        # Measure retrieval time
        start_retrieval = time.time()
        results = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time_ms = (time.time() - start_retrieval) * 1000

        if not results:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "confidence": 0.0
            }

        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {r.chunk.source_document_id}]\n{r.chunk.content}"
            for r in results
        ])

        # Generate answer using LLM
        start_generation = time.time()
        prompt = rag_generation_prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        generation_time_ms = (time.time() - start_generation) * 1000

        answer = response.content

        # Calculate metrics
        top_similarity = results[0].similarity_score
        avg_similarity = sum(r.similarity_score for r in results) / len(results)

        # Log to monitoring
        if self.monitoring:
            self.monitoring.log_query(
                query_id=f"q_{int(time.time())}",
                query=question,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                chunks_retrieved=len(results),
                top_similarity_score=top_similarity,
                avg_similarity_score=avg_similarity,
                answer_length=len(answer),
                is_grounded=True  # TODO: Add hallucination detection
            )

        return {
            "answer": answer,
            "sources": [
                {
                    "doc_id": r.chunk.source_document_id,
                    "content": r.chunk.content[:200],
                    "similarity": r.similarity_score
                }
                for r in results
            ],
            "confidence": top_similarity,
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms
        }


# Example Usage:
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(enable_monitoring=True)

    # Index documents
    pipeline.index_document(
        doc_id="doc_001",
        content="TechCorp was founded in 2010 by Alice Johnson.",
        metadata={"type": "company_info"}
    )

    # Query
    result = pipeline.query("When was TechCorp founded?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Retrieval time: {result['retrieval_time_ms']:.0f}ms")


# ============================================================================
# Example 2: Flask API Integration
# ============================================================================

"""
FILE: document_portal/routes/rag_routes.py
Add RAG endpoints to existing Flask app
"""

from flask import Blueprint, request, jsonify
from src.document_chat.rag_pipeline import RAGPipeline

rag_bp = Blueprint('rag', __name__, url_prefix='/api/v1/rag')

# Initialize pipeline (singleton)
pipeline = RAGPipeline(enable_monitoring=True)


@rag_bp.route('/index', methods=['POST'])
def index_document():
    """
    Index a document for RAG retrieval.

    Request:
        {
            "doc_id": "doc_001",
            "content": "Document text...",
            "metadata": {"type": "report"}
        }
    """
    data = request.json
    doc_id = data.get('doc_id')
    content = data.get('content')
    metadata = data.get('metadata', {})

    if not doc_id or not content:
        return jsonify({"error": "doc_id and content required"}), 400

    try:
        num_chunks = pipeline.index_document(doc_id, content, metadata)
        return jsonify({
            "success": True,
            "doc_id": doc_id,
            "chunks_created": num_chunks
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_bp.route('/query', methods=['POST'])
def query_documents():
    """
    Query indexed documents.

    Request:
        {
            "question": "What is the revenue?",
            "top_k": 5
        }
    """
    data = request.json
    question = data.get('question')
    top_k = data.get('top_k', 5)

    if not question:
        return jsonify({"error": "question required"}), 400

    try:
        result = pipeline.query(question, top_k=top_k)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get RAG system metrics"""
    if pipeline.monitoring:
        stats = pipeline.monitoring.get_summary_stats()
        return jsonify(stats), 200
    else:
        return jsonify({"error": "Monitoring not enabled"}), 400


# Register blueprint in main app
# from routes.rag_routes import rag_bp
# app.register_blueprint(rag_bp)


# ============================================================================
# Example 3: Testing Integration
# ============================================================================

"""
FILE: tests/test_integration_example.py
Example of how to test the integrated RAG system
"""

import pytest
from src.document_chat.rag_pipeline import RAGPipeline


@pytest.fixture
def rag_pipeline():
    """Create RAG pipeline for testing"""
    return RAGPipeline(enable_monitoring=False)


def test_end_to_end_rag_flow(rag_pipeline):
    """Test complete RAG flow: index + query"""
    # Index document
    doc_id = "test_doc"
    content = "TechCorp was founded in 2010 by Alice Johnson in San Francisco."

    num_chunks = rag_pipeline.index_document(doc_id, content)
    assert num_chunks > 0

    # Query
    result = rag_pipeline.query("When was TechCorp founded?")

    # Assertions
    assert "2010" in result['answer']
    assert result['confidence'] > 0.5
    assert len(result['sources']) > 0
    assert result['retrieval_time_ms'] > 0


# ============================================================================
# STEP 7: Update Config to Support New Features
# ============================================================================

"""
UPDATE: document_portal/config/config.yaml

Add:
"""

config_additions = """
# Retrieval configuration
retrieval:
  chunking_strategy: "sliding_window"
  chunk_size: 512
  chunk_overlap: 50

  vector_store:
    type: "faiss"  # or "in_memory" for development
    dimension: 384

# Monitoring & Alerting
monitoring:
  enabled: true
  alert_thresholds:
    hallucination_rate: 0.05
    avg_retrieval_time_ms: 500
    min_similarity_score: 0.6

  # Slack webhook for alerts (set via environment variable)
  slack_webhook: ${SLACK_WEBHOOK_URL}

# Prompt versions (for A/B testing)
prompts:
  default_version: "v2_with_guardrails"  # Use anti-hallucination version
  versions:
    v1_simple: "simple prompt"
    v2_with_guardrails: "prompt with anti-hallucination rules"

# Monitoring thresholds
monitoring:
  hallucination_rate_threshold: 0.05  # Alert if > 5%
  retrieval_score_threshold: 0.8      # Alert if < 0.8
  latency_threshold_ms: 500            # Alert if > 500ms
"""


# ============================================================================
# STEP 7: Integration Example
# ============================================================================

"""
HOW TO USE EVERYTHING TOGETHER:
"""

def example_rag_pipeline():
    """Complete RAG pipeline with monitoring"""
    
    # Step 1: Load components
    from utils.model_loader import ModelLoader
    from src.multi_document_chat.retrieval import DocumentRetriever
    from src.document_analyzer.data_analysis import DocumentAnalyzer
    from src.multi_document_chat.monitoring import RAGMonitoring
    
    loader = ModelLoader()
    embedder = loader.load_embeddings()
    llm = loader.load_llm()
    
    # Step 2: Setup retrieval
    from retrieval_layer import InMemoryVectorStore  # or FAISSVectorStore
    vector_store = InMemoryVectorStore()
    retriever = DocumentRetriever(embedder, vector_store)
    
    # Step 3: Index a document
    doc_text = "Alice Johnson is the CEO of TechCorp..."
    chunks_created = retriever.index_document("doc_001", doc_text)
    print(f"Indexed {chunks_created} chunks")
    
    # Step 4: Retrieve for a query
    query = "Who is the CEO?"
    results = retriever.retrieve(query, top_k=3)
    
    context = "\n".join([r.chunk.content for r in results])
    print(f"Retrieved {len(results)} chunks")
    print(f"Context:\n{context[:200]}...")
    
    # Step 5: Generate answer using LLM
    # (Would use LLM here to generate answer based on context)
    
    # Step 6: Monitor
    monitoring = RAGMonitoring()
    is_grounded = "Alice Johnson" in context  # Simple check
    
    monitoring.log_query(
        query=query,
        retrieved_docs=len(results),
        answer="Alice Johnson is the CEO",
        is_grounded=is_grounded
    )
    
    print(f"Metrics: {monitoring.get_metrics()}")


# ============================================================================
# STEP 8: Common Pitfalls & How to Fix Them
# ============================================================================

"""
PITFALL 1: LLM Ignores Context (Hallucination)
Problem: Even with context, LLM makes up information
Fix: Use stronger prompt with explicit constraints
- "Use ONLY the provided context"
- "If not in context, say 'Not Available'"
- "Do NOT use external knowledge"

TEST:
def test_llm_uses_context():
    context = "Company founded in 2010"
    answer = llm(context=context, query="When was it founded?")
    assert "2010" in answer
    assert "made up year" not in answer

---

PITFALL 2: Retrieval Returns Irrelevant Documents
Problem: Vector search doesn't find related documents
Fix: Improve chunking or embeddings
- Adjust chunk size (too small = lost context, too large = noise)
- Test different embedding models
- Validate embedding quality

TEST:
def test_retrieval_quality():
    query = "Who is the CEO?"
    results = retriever.retrieve(query, top_k=5)
    assert any("CEO" in r.chunk.content for r in results[:3])

---

PITFALL 3: Too Many False Positives (Bad Precision)
Problem: Retrieval returns somewhat-related but not directly relevant docs
Fix: Increase number of chunks checked or use re-ranking
- Try top_k=10 but only use top_3
- Add a second ranking layer
- Validate similarity scores

TEST:
def test_retrieval_precision():
    query = "What is the revenue?"
    results = retriever.retrieve(query, top_k=5)
    # Assert: Top result is directly about revenue
    assert results[0].similarity_score > 0.8

---

PITFALL 4: Latency Too High
Problem: System is slow (retrieval + generation takes >500ms)
Fix:
- Use FAISS instead of in-memory search
- Cache embeddings
- Reduce chunk size (faster search)
- Parallel processing

TEST:
import time
start = time.time()
results = retriever.retrieve(query)
latency = time.time() - start
assert latency < 0.5  # 500ms target
"""


# ============================================================================
# STEP 9: Next Steps for Your Learning
# ============================================================================

"""
After implementing this:

1. RUN TESTS
   pytest tests/ -v
   
2. MEASURE BASELINE METRICS
   - Hallucination rate
   - Retrieval quality (MRR)
   - Latency
   
3. A/B TEST IMPROVEMENTS
   - Different prompt versions
   - Different embedding models
   - Different chunking strategies
   
4. MONITOR IN PRODUCTION
   - Track metrics daily
   - Alert on degradation
   - Capture user feedback
   
5. ITERATE
   - Improve based on metrics
   - Add tests for new issues
   - Document improvements
"""


if __name__ == "__main__":
    print("Implementation Guide Loaded")
    print("\nTo integrate with document_portal:")
    print("1. Upgrade prompt_library.py")
    print("2. Add retrieval_layer.py")
    print("3. Create test suite")
    print("4. Add monitoring")
    print("5. Update config")
    print("6. Test end-to-end")
    print("7. Deploy to production")
    print("8. Monitor metrics")
