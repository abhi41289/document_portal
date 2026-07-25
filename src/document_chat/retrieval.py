import sys
import os
import time
from datetime import datetime, timedelta
from operator import itemgetter
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from src.document_chat.rag_monitoring import RAGMonitoring
from src.document_chat.tracing import RAGInvocationTrace


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None, monitoring: Optional[RAGMonitoring] = None):
        try:
            self.session_id = session_id

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.vectorstore = None
            self.retrieval_k = 5
            self.monitoring = monitoring or RAGMonitoring()
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )
            self.vectorstore = vectorstore
            self.retrieval_k = k

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the conversational RAG pipeline and return only the answer."""
        answer, _ = self.invoke_with_trace(user_input=user_input, chat_history=chat_history)
        return answer

    def invoke_with_trace(
        self,
        user_input: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Invoke the RAG pipeline and return both the answer and a structured trace."""
        trace = RAGInvocationTrace(question=user_input, session_id=self.session_id)
        try:
            if self.retriever is None and self.vectorstore is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().",
                    sys,
                )

            chat_history = chat_history or []
            chat_history_text = self._format_chat_history(chat_history)

            rewrite_started = time.perf_counter()
            rewrite_prompt = self.contextualize_prompt.format(
                input=user_input,
                chat_history=chat_history_text,
            )
            rewrite_raw = self.llm.invoke(rewrite_prompt)
            rewritten_question = self._extract_text(rewrite_raw).strip() or user_input
            rewrite_ended = time.perf_counter()
            rewrite_started_at, rewrite_ended_at = self._stage_timestamps(
                rewrite_started, rewrite_ended
            )
            trace.add_stage(
                name="rewrite",
                request={
                    "input": user_input,
                    "chat_history": chat_history_text,
                },
                response={
                    "prompt": rewrite_prompt,
                    "rewritten_question": rewritten_question,
                    "raw_response": self._extract_text(rewrite_raw),
                },
                started_at=rewrite_started_at,
                ended_at=rewrite_ended_at,
            )

            retrieval_started = time.perf_counter()
            retrieved_docs, retrieval_scores = self._retrieve_documents(rewritten_question)
            retrieval_ended = time.perf_counter()
            retrieval_started_at, retrieval_ended_at = self._stage_timestamps(
                retrieval_started, retrieval_ended
            )
            context = self._format_docs(retrieved_docs)
            top_similarity_score = max(retrieval_scores) if retrieval_scores else 0.0
            avg_similarity_score = (
                sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
            )
            trace.add_stage(
                name="retrieval",
                request={
                    "query": rewritten_question,
                    "top_k": self.retrieval_k,
                },
                response={
                    "retrieved_count": len(retrieved_docs),
                    "top_similarity_score": top_similarity_score,
                    "avg_similarity_score": avg_similarity_score,
                    "context": context,
                    "retrieved_documents": [
                        self._serialize_document(doc) for doc in retrieved_docs
                    ],
                },
                started_at=retrieval_started_at,
                ended_at=retrieval_ended_at,
            )

            generation_started = time.perf_counter()
            generation_prompt = self.qa_prompt.format(
                context=context,
                input=user_input,
                chat_history=chat_history_text,
            )
            generation_raw = self.llm.invoke(generation_prompt)
            answer = self._extract_text(generation_raw).strip()
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                answer = "no answer generated."
            generation_ended = time.perf_counter()
            generation_started_at, generation_ended_at = self._stage_timestamps(
                generation_started, generation_ended
            )
            trace.add_stage(
                name="generation",
                request={
                    "input": user_input,
                    "context": context,
                    "chat_history": chat_history_text,
                },
                response={
                    "prompt": generation_prompt,
                    "answer": answer,
                    "raw_response": self._extract_text(generation_raw),
                    "answer_preview": answer[:250],
                },
                started_at=generation_started_at,
                ended_at=generation_ended_at,
            )

            total_time_ms = (
                time.perf_counter() - rewrite_started
            ) * 1000.0
            trace.succeed(
                total_time_ms=total_time_ms,
                answer_length=len(answer),
                top_similarity_score=top_similarity_score,
                avg_similarity_score=avg_similarity_score,
                retrieved_count=len(retrieved_docs),
            )

            if self.monitoring is not None:
                self.monitoring.log_query(
                    query_id=trace.trace_id,
                    query=user_input,
                    retrieval_time_ms=(retrieval_ended - retrieval_started) * 1000.0,
                    generation_time_ms=(generation_ended - generation_started) * 1000.0,
                    chunks_retrieved=len(retrieved_docs),
                    top_similarity_score=top_similarity_score,
                    avg_similarity_score=avg_similarity_score,
                    answer_length=len(answer),
                    hallucination_score=None,
                    is_grounded=None,
                )

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=answer[:150],
                trace_id=trace.trace_id,
            )
            return answer, trace.to_dict()
        except Exception as e:
            trace.fail(
                reason=self._classify_failure(e),
                error_message=str(e),
            )
            if self.monitoring is not None:
                self.monitoring.log_error(
                    trace.trace_id,
                    self._classify_failure(e),
                    str(e),
                )
            wrapped = DocumentPortalException("Invocation error in ConversationalRAG", e)
            setattr(wrapped, "trace", trace.to_dict())
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise wrapped

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if response is None:
            return ""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    @staticmethod
    def _format_chat_history(chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return ""
        formatted = []
        for message in chat_history:
            role = getattr(message, "type", getattr(message, "role", "message"))
            content = getattr(message, "content", str(message))
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    @staticmethod
    def _serialize_document(document: Any) -> Dict[str, Any]:
        return {
            "content": getattr(document, "page_content", getattr(document, "content", str(document))),
            "metadata": getattr(document, "metadata", {}),
        }

    @staticmethod
    def _stage_timestamps(start_perf: float, end_perf: float) -> Tuple[str, str]:
        started_at = datetime.utcnow()
        ended_at = started_at + timedelta(milliseconds=max((end_perf - start_perf) * 1000.0, 0.0))
        return started_at.isoformat(), ended_at.isoformat()

    @staticmethod
    def _classify_failure(error: Exception) -> str:
        if isinstance(error, DocumentPortalException):
            return "document_portal_exception"
        return error.__class__.__name__.lower()

    def _retrieve_documents(self, query: str) -> Tuple[List[Any], List[float]]:
        top_k = self.retrieval_k
        if self.vectorstore is not None:
            if hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
                scored_docs = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
                documents = [doc for doc, _ in scored_docs]
                scores = [float(score) for _, score in scored_docs]
                return documents, scores
            if hasattr(self.vectorstore, "similarity_search_with_score"):
                scored_docs = self.vectorstore.similarity_search_with_score(query, k=top_k)
                documents = [doc for doc, _ in scored_docs]
                scores = [1.0 / (1.0 + float(score)) for _, score in scored_docs]
                return documents, scores

        if self.retriever is None:
            return [], []

        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
        elif hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(query)
        else:
            docs = []

        if docs is None:
            docs = []
        if not isinstance(docs, list):
            docs = list(docs)
        return docs, []

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
