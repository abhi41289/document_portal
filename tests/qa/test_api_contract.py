"""API contract tests for the chat trace payload."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest


def _run_contract_probe(mode: str) -> dict:
    script = textwrap.dedent(
        f"""
        import json
        import sys
        import types

        def add_module(name):
            module = types.ModuleType(name)
            sys.modules[name] = module
            return module

        # External dependency stubs used during import of api.main and its transitive modules.
        fitz = add_module("fitz")
        fitz.open = lambda *args, **kwargs: None

        dotenv = add_module("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None

        pandas = add_module("pandas")
        class DataFrame:
            def __init__(self, rows=None):
                self.rows = list(rows or [])
            def to_dict(self, orient="records"):
                return list(self.rows)
        pandas.DataFrame = DataFrame

        langchain = add_module("langchain")
        langchain_schema = add_module("langchain.schema")
        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {{}}
        langchain_schema.Document = Document
        langchain.schema = langchain_schema

        langchain_output = add_module("langchain.output_parsers")
        class OutputFixingParser:
            @classmethod
            def from_llm(cls, parser=None, llm=None):
                return cls()
        langchain_output.OutputFixingParser = OutputFixingParser
        langchain.output_parsers = langchain_output

        langchain_google_genai = add_module("langchain_google_genai")
        class GoogleGenerativeAIEmbeddings:
            def __init__(self, *args, **kwargs):
                pass
        class ChatGoogleGenerativeAI:
            def __init__(self, *args, **kwargs):
                pass
        langchain_google_genai.GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings
        langchain_google_genai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI

        langchain_groq = add_module("langchain_groq")
        class ChatGroq:
            def __init__(self, *args, **kwargs):
                pass
        langchain_groq.ChatGroq = ChatGroq

        langchain_core = add_module("langchain_core")
        langchain_core_messages = add_module("langchain_core.messages")
        class BaseMessage:
            pass
        langchain_core_messages.BaseMessage = BaseMessage
        langchain_core.messages = langchain_core_messages

        langchain_core_prompts = add_module("langchain_core.prompts")
        class _PromptTemplate:
            def __init__(self, template=""):
                self.template = template
            @classmethod
            def from_template(cls, template):
                return cls(template)
            def format(self, **kwargs):
                return self.template.format(**kwargs)
            def __or__(self, other):
                return self
        langchain_core_prompts.ChatPromptTemplate = _PromptTemplate
        langchain_core_prompts.PromptTemplate = _PromptTemplate
        langchain_core.prompts = langchain_core_prompts

        langchain_core_output = add_module("langchain_core.output_parsers")
        class JsonOutputParser:
            def __init__(self, *args, **kwargs):
                pass
            def get_format_instructions(self):
                return "{{}}"
        class StrOutputParser:
            def __or__(self, other):
                return self
        langchain_core_output.JsonOutputParser = JsonOutputParser
        langchain_core_output.StrOutputParser = StrOutputParser
        langchain_core.output_parsers = langchain_core_output

        langchain_community = add_module("langchain_community")
        langchain_community_vectorstores = add_module("langchain_community.vectorstores")
        class FAISS:
            @classmethod
            def load_local(cls, *args, **kwargs):
                return cls()
            @classmethod
            def from_texts(cls, *args, **kwargs):
                return cls()
            def save_local(self, *args, **kwargs):
                return None
            def as_retriever(self, *args, **kwargs):
                return self
            def similarity_search_with_relevance_scores(self, *args, **kwargs):
                return []
            def similarity_search_with_score(self, *args, **kwargs):
                return []
        langchain_community_vectorstores.FAISS = FAISS
        langchain_community.vectorstores = langchain_community_vectorstores

        langchain_community_loaders = add_module("langchain_community.document_loaders")
        class PyPDFLoader:
            def __init__(self, *args, **kwargs):
                pass
            def load(self):
                return []
        class Docx2txtLoader(PyPDFLoader):
            pass
        class TextLoader(PyPDFLoader):
            pass
        langchain_community_loaders.PyPDFLoader = PyPDFLoader
        langchain_community_loaders.Docx2txtLoader = Docx2txtLoader
        langchain_community_loaders.TextLoader = TextLoader
        langchain_community.document_loaders = langchain_community_loaders

        langchain_text_splitters = add_module("langchain_text_splitters")
        class RecursiveCharacterTextSplitter:
            def __init__(self, *args, **kwargs):
                pass
            def split_documents(self, docs):
                return docs
        langchain_text_splitters.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

        import api.main as api_main
        from exception.custom_exception import DocumentPortalException
        from fastapi.testclient import TestClient

        class SuccessRAG:
            def __init__(self, session_id=None, retriever=None):
                self.session_id = session_id
            def load_retriever_from_faiss(self, *args, **kwargs):
                return self
            def invoke_with_trace(self, question, chat_history=None):
                return (
                    "Alice Johnson is the CEO of TechCorp.",
                    {{
                        "trace_id": "trace-success",
                        "session_id": self.session_id,
                        "status": "success",
                        "stages": [
                            {{"name": "rewrite", "response": {{"rewritten_question": question}}}},
                            {{
                                "name": "retrieval",
                                "response": {{
                                    "retrieved_documents": [
                                        {{
                                            "content": "Alice Johnson is the CEO of TechCorp.",
                                            "metadata": {{"source_document_id": "doc_company"}}
                                        }}
                                    ]
                                }}
                            }},
                            {{"name": "generation", "response": {{"answer": "Alice Johnson is the CEO of TechCorp."}}}}
                        ]
                    }}
                )

        class FailureRAG:
            def __init__(self, session_id=None, retriever=None):
                self.session_id = session_id
            def load_retriever_from_faiss(self, *args, **kwargs):
                return self
            def invoke_with_trace(self, question, chat_history=None):
                error = DocumentPortalException("Invocation error in ConversationalRAG", Exception("boom"))
                error.trace = {{
                    "trace_id": "trace-error",
                    "session_id": self.session_id,
                    "status": "error",
                    "failure_reason": "valueerror",
                    "stages": []
                }}
                raise error

        api_main.os.path.isdir = lambda path: True
        api_main.ConversationalRAG = SuccessRAG if {mode!r} == "success" else FailureRAG
        client = TestClient(api_main.app)

        if {mode!r} == "success":
            response = client.post(
                "/chat/query",
                data={{"question": "Who is the CEO?", "session_id": "session-1", "use_session_dirs": "true", "k": "3"}},
            )
            payload = {{"status_code": response.status_code, "json": response.json()}}
        else:
            response = client.post(
                "/chat/query",
                data={{"question": "Who is the CEO?", "session_id": "session-1", "use_session_dirs": "true", "k": "3"}},
            )
            payload = {{"status_code": response.status_code, "json": response.json()}}

        print(json.dumps(payload))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip())


@pytest.mark.integration
def test_chat_query_returns_trace_payload():
    result = _run_contract_probe("success")

    assert result["status_code"] == 200
    assert result["json"]["answer"] == "Alice Johnson is the CEO of TechCorp."
    assert result["json"]["trace"]["status"] == "success"
    assert result["json"]["trace"]["stages"][1]["response"]["retrieved_documents"][0]["metadata"]["source_document_id"] == "doc_company"


@pytest.mark.integration
def test_chat_query_returns_structured_failure_trace():
    result = _run_contract_probe("failure")

    assert result["status_code"] == 500
    assert result["json"]["detail"]["trace"]["status"] == "error"
    assert result["json"]["detail"]["trace"]["failure_reason"] == "valueerror"

