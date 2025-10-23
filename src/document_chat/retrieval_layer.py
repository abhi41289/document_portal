"""
Complete Retrieval Layer for document_portal
Implements vector search, chunking, and embedding management with full testability
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    source_document_id: str = None
    chunk_index: int = None
    

@dataclass
class RetrievalResult:
    """Result of a retrieval query"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int


# ============================================================================
# ABSTRACT INTERFACES (for testability and extensibility)
# ============================================================================

class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embedding vectors"""
        pass


class GoogleGenAIEmbeddingProvider(EmbeddingProvider):
    """
    Adapter for Google Generative AI Embeddings (from utils.model_loader.ModelLoader)
    Bridges existing document_portal embeddings with the new retrieval layer
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: GoogleGenerativeAIEmbeddings instance from ModelLoader.load_embeddings()
        """
        self.embedder = embedder
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized GoogleGenAI embedding provider adapter")

    def embed_text(self, text: str) -> List[float]:
        """Convert single text to embedding vector"""
        try:
            embedding = self.embedder.embed_query(text)
            self.logger.debug(f"Embedded text of length {len(text)}, vector dim: {len(embedding)}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error embedding text: {str(e)}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embedding vectors"""
        try:
            embeddings = self.embedder.embed_documents(texts)
            self.logger.info(f"Embedded batch of {len(texts)} texts")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error embedding batch: {str(e)}")
            raise


class VectorStore(ABC):
    """Abstract interface for vector storage"""
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete all chunks from a specific document"""
        pass


# ============================================================================
# CHUNKING STRATEGIES
# ============================================================================

class ChunkingStrategy(ABC):
    """Abstract base for different chunking approaches"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document into chunks"""
        pass


class SlidingWindowChunker(ChunkingStrategy):
    """
    Split text into chunks with sliding window overlap.
    This is the standard approach for RAG systems.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, doc_id: str = ""):
        """
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
            doc_id: Source document ID
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.doc_id = doc_id
        self.logger = logging.getLogger(__name__)
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create overlapping chunks from text.
        
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        if not text or len(text) == 0:
            self.logger.warning(f"Empty text provided for document {self.doc_id}")
            return chunks
        
        # Create chunks with overlap
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Skip very small chunks (likely incomplete)
            if len(chunk_text.strip()) < 50:
                continue
            
            chunk = DocumentChunk(
                id=f"{self.doc_id}_chunk_{len(chunks)}",
                content=chunk_text,
                metadata={**metadata, "chunk_start": i, "chunk_end": i + len(chunk_text)},
                source_document_id=self.doc_id,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks from document {self.doc_id}")
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Split text by semantic boundaries (sentences, paragraphs).
    Better for maintaining context but requires more computation.
    """
    
    def __init__(self, doc_id: str = "", target_size: int = 512):
        self.doc_id = doc_id
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into semantically coherent chunks.
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds target size, save current chunk
            if len(current_chunk) + len(para) > self.target_size and current_chunk:
                chunk = DocumentChunk(
                    id=f"{self.doc_id}_chunk_{len(chunks)}",
                    content=current_chunk.strip(),
                    metadata={**metadata, "semantic_unit": "paragraph_group"},
                    source_document_id=self.doc_id,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
                current_chunk = ""
            
            current_chunk += "\n\n" + para
        
        # Add remaining content
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=f"{self.doc_id}_chunk_{len(chunks)}",
                content=current_chunk.strip(),
                metadata={**metadata, "semantic_unit": "final"},
                source_document_id=self.doc_id,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks


# ============================================================================
# IN-MEMORY VECTOR STORE (for development/testing)
# ============================================================================

class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store for testing.
    Use for development, unit tests, and small datasets.
    """
    
    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to the store"""
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            self.chunks[chunk.id] = chunk
        
        self.logger.info(f"Added {len(chunks)} chunks. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """
        Search using cosine similarity.
        """
        if not self.chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk_id, chunk in self.chunks.items():
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk_id, similarity) in enumerate(similarities[:top_k]):
            results.append(RetrievalResult(
                chunk=self.chunks[chunk_id],
                similarity_score=similarity,
                rank=rank + 1
            ))
        
        return results
    
    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete all chunks from a document"""
        to_delete = [cid for cid, chunk in self.chunks.items() 
                     if chunk.source_document_id == doc_id]
        for cid in to_delete:
            del self.chunks[cid]
        
        self.logger.info(f"Deleted {len(to_delete)} chunks from document {doc_id}")
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if magnitude == 0:
            return 0.0
        
        return float(dot_product / magnitude)


# ============================================================================
# PRODUCTION VECTOR STORE (FAISS)
# ============================================================================

class FAISSVectorStore(VectorStore):
    """
    Production vector store using FAISS (Facebook AI Similarity Search).
    Faster than in-memory for large datasets.
    """
    
    def __init__(self):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings = []
        self.index = None
        self.logger = logging.getLogger(__name__)
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks and build FAISS index"""
        embeddings = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            
            self.chunks[chunk.id] = chunk
            embeddings.append(chunk.embedding)
        
        if embeddings:
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = self.faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            self.embeddings = embeddings
            self.logger.info(f"Built FAISS index with {len(embeddings)} vectors")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """Search using FAISS"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        chunk_ids = list(self.chunks.keys())
        
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            chunk_id = chunk_ids[idx]
            # Convert L2 distance to similarity (0-1 scale)
            similarity = 1.0 / (1.0 + distance)
            
            results.append(RetrievalResult(
                chunk=self.chunks[chunk_id],
                similarity_score=similarity,
                rank=rank + 1
            ))
        
        return results
    
    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete chunks (requires rebuilding index)"""
        to_delete = [cid for cid, chunk in self.chunks.items() 
                     if chunk.source_document_id == doc_id]
        
        for cid in to_delete:
            del self.chunks[cid]
        
        # Rebuild index (expensive, but necessary)
        self.index = None
        self.embeddings = []
        
        if self.chunks:
            chunks = list(self.chunks.values())
            self.add_chunks(chunks)
        
        self.logger.info(f"Deleted {len(to_delete)} chunks from document {doc_id}")


# ============================================================================
# RETRIEVER: Orchestrates chunking, embedding, and search
# ============================================================================

class DocumentRetriever:
    """
    Main retriever that orchestrates the full pipeline:
    1. Chunk documents
    2. Embed chunks
    3. Store in vector DB
    4. Retrieve similar chunks for queries
    """
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        chunking_strategy: ChunkingStrategy = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunking_strategy = chunking_strategy or SlidingWindowChunker()
        self.logger = logging.getLogger(__name__)
        self.session_id = f"retriever_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def index_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        Index a document: chunk it, embed it, and store it.
        
        Returns:
            Number of chunks created
        """
        if metadata is None:
            metadata = {}
        
        # Step 1: Chunk the document
        self.chunking_strategy.doc_id = doc_id
        chunks = self.chunking_strategy.chunk(text, metadata)
        
        if not chunks:
            self.logger.warning(f"No chunks created for document {doc_id}")
            return 0
        
        # Step 2: Embed chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Step 3: Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Step 4: Store in vector DB
        self.vector_store.add_chunks(chunks)
        
        self.logger.info(
            f"Indexed document {doc_id}: {len(chunks)} chunks, "
            f"session {self.session_id}"
        )
        
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve similar chunks for a query.
        
        Returns:
            List of RetrievalResult objects ranked by similarity
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        self.logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:50]}...', "
            f"session {self.session_id}"
        )
        
        return results
    
    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index"""
        self.vector_store.delete_by_document_id(doc_id)
        self.logger.info(f"Removed document {doc_id}")


# ============================================================================
# RETRIEVAL QUALITY METRICS (for testing)
# ============================================================================

class RetrievalMetrics:
    """Calculate metrics for retrieval quality"""
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ranks: List[int]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        Measures position of first relevant result
        """
        if not retrieved_ranks:
            return 0.0
        return np.mean([1.0 / rank for rank in retrieved_ranks])
    
    @staticmethod
    def precision_at_k(relevant_count: int, k: int) -> float:
        """
        Precision@K
        What fraction of top-k results are relevant?
        """
        if k == 0:
            return 0.0
        return relevant_count / k
    
    @staticmethod
    def similarity_distribution(similarities: List[float]) -> Dict[str, float]:
        """
        Analyze distribution of similarity scores
        Helps diagnose if retrieval is working
        """
        similarities = np.array(similarities)
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities))
        }


if __name__ == "__main__":
    print("Retrieval Layer Module Loaded")
    print("Available components:")
    print("  - ChunkingStrategy (abstract)")
    print("    - SlidingWindowChunker")
    print("    - SemanticChunker")
    print("  - VectorStore (abstract)")
    print("    - InMemoryVectorStore (for testing)")
    print("    - FAISSVectorStore (for production)")
    print("  - DocumentRetriever (main orchestrator)")
    print("  - RetrievalMetrics (quality measurement)")
