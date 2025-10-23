"""
Enhanced Prompt Library for document_portal
Includes anti-hallucination guardrails and better constraints
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# ============================================================================
# METADATA EXTRACTION PROMPT (for data_analyzer)
# ============================================================================

metadata_extraction_prompt = ChatPromptTemplate.from_template("""
You are a document metadata analyzer. Your ONLY job is to extract structured information 
that EXPLICITLY appears in the provided document.

CRITICAL RULES:
1. Extract ONLY information that is directly stated in the document
2. Do NOT infer, guess, or use external knowledge
3. For missing information, use these exact placeholders:
   - Unknown/Not Provided fields → "Not Available"
   - Date fields with no date → "Not Available"
   - Numeric fields that are unclear → "Not Available"
4. If unsure whether information is in the document, mark as "Not Available"
5. ALWAYS return valid JSON matching the schema below

DOCUMENT METADATA SCHEMA:
{format_instructions}

DOCUMENT TO ANALYZE:
---BEGIN DOCUMENT---
{document_text}
---END DOCUMENT---

EXTRACTION INSTRUCTIONS:
- Title: Extract the document's main title/heading if present. Otherwise: "Not Available"
- Author: Extract author name if explicitly stated. Do NOT guess. Otherwise: "Unknown"
- DateCreated: Extract creation/publication date if stated. Otherwise: "Not Available"
- LastModifiedDate: Extract last modified date if present. Otherwise: "Not Available"
- Publisher: Extract publisher name if stated. Otherwise: "Not Available"
- Language: Detect language of document. Otherwise: "Not Detected"
- PageCount: Extract if stated, or estimate from content. If unsure: "Not Available"
- SentimentTone: Analyze tone (Professional, Academic, Casual, Formal, etc.)
- Summary: Create 2-3 bullet points of key information from document ONLY

REMEMBER: Every field must be supported by the actual document text.
If the document doesn't contain information, use "Not Available".
Do NOT add information from your training data.

Return ONLY valid JSON, no additional text:
""")


# ============================================================================
# RAG GENERATION PROMPT (for multi_document_chat)
# ============================================================================

rag_generation_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based ONLY on provided documents.

STRICT RULES:
1. Answer questions using ONLY the information provided in the context below
2. If the answer is not in the context, say: "This information is not available in the provided documents."
3. Do NOT use your general knowledge or training data
4. If documents contain conflicting information, mention the conflict
5. Always cite which document the information comes from
6. Be precise and factual

PROVIDED CONTEXT:
---BEGIN CONTEXT---
{context}
---END CONTEXT---

USER QUESTION:
{question}

INSTRUCTIONS FOR RESPONSE:
- If answer is in context: Provide clear, concise answer with document references
- If answer is NOT in context: Say "Not available in provided documents"
- If context has conflicting info: Acknowledge the conflict
- Do NOT guess or make up information

Your response:
""")


# ============================================================================
# COMPARISON PROMPT (for document_compare)
# ============================================================================

document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a document comparison specialist. Compare the two documents based ONLY on 
information explicitly stated in them.

DOCUMENTS TO COMPARE:
---DOCUMENT 1---
{document_1}
---DOCUMENT 2---
{document_2}

COMPARISON CRITERIA:
{comparison_criteria}

RULES:
1. Only compare information that appears in BOTH documents
2. Note what information is in Doc1 but missing in Doc2 (and vice versa)
3. Highlight contradictions if they exist
4. Do NOT infer information that isn't explicitly stated
5. Be objective and factual

Provide structured comparison:
""")


# ============================================================================
# EVALUATION PROMPT (for testing hallucinations)
# ============================================================================

hallucination_detection_prompt = ChatPromptTemplate.from_template("""
You are a hallucination detector for RAG systems.
Your job is to check if a generated answer is grounded in the provided context.

CONTEXT (what the RAG system had to work with):
---BEGIN CONTEXT---
{context}
---END CONTEXT---

GENERATED ANSWER (what the RAG system produced):
---BEGIN ANSWER---
{answer}
---END ANSWER---

ORIGINAL QUESTION:
{question}

EVALUATION CRITERIA:
1. Is each claim in the answer supported by the context?
2. Does the answer use external knowledge not in context?
3. Are there contradictions with the context?
4. Is the answer consistent with what the context says?

Provide evaluation in JSON format:
{{
    "is_grounded": true/false,
    "hallucination_score": 0.0-1.0,
    "supported_claims": ["claim1", "claim2"],
    "unsupported_claims": ["claim3", "claim4"],
    "external_knowledge_used": ["knowledge1"],
    "explanation": "Detailed explanation"
}}

Evaluation:
""")


# ============================================================================
# GUARDRAIL PROMPT (catches attempts to bypass constraints)
# ============================================================================

guardrail_prompt = ChatPromptTemplate.from_template("""
You are a guardrail system for RAG applications.
Check if a response violates these constraints:

CONSTRAINT 1: No External Knowledge
- Response should use ONLY provided context
- Should NOT supplement from training data

CONSTRAINT 2: Explicit Uncertainty
- If unsure, should say "Not available" or "I don't know"
- Should NOT guess or make up information

CONSTRAINT 3: Source Attribution
- Should cite which document information comes from
- Should NOT claim certainty about unsourced info

RESPONSE TO CHECK:
{response}

PROVIDED CONTEXT:
{context}

Check for constraint violations and return JSON:
{{
    "violates_external_knowledge": true/false,
    "violates_uncertainty_handling": true/false,
    "violates_attribution": true/false,
    "severity": "critical/warning/info",
    "remediation": "How to fix the response"
}}

Analysis:
""")


# ============================================================================
# UTILITY: Generate Prompt with Context
# ============================================================================

def create_rag_generation_prompt(context: str, question: str) -> str:
    """
    Helper to create a complete RAG prompt with context.
    This ensures consistent prompting across the application.
    """
    return rag_generation_prompt.format(context=context, question=question)


def create_metadata_extraction_prompt(document_text: str, format_instructions: str) -> str:
    """
    Helper to create metadata extraction prompt.
    """
    return metadata_extraction_prompt.format(
        document_text=document_text,
        format_instructions=format_instructions
    )


# ============================================================================
# CONFIGURATION: Prompt Versions (for A/B testing)
# ============================================================================

PROMPT_VERSIONS = {
    "v1_original": {
        "name": "Original Simple Prompt",
        "template": ChatPromptTemplate.from_template("""
            You are a helpful assistant.
            {format_instructions}
            Analyze this document:
            {document_text}
        """),
        "guardrails": False,
    },
    "v2_enhanced": {
        "name": "Enhanced with Guardrails",
        "template": metadata_extraction_prompt,
        "guardrails": True,
    },
}


# ============================================================================
# TESTING: Hallucination Test Cases
# ============================================================================

HALLUCINATION_TEST_CASES = [
    {
        "name": "Direct Answer in Context",
        "question": "Who is the CEO?",
        "context": "Alice Johnson serves as the CEO of the company.",
        "expected_answer": "Alice Johnson",
        "should_NOT_contain": ["Bob", "Smith", "unknown"],
    },
    {
        "name": "Answer NOT in Context",
        "question": "What is the revenue?",
        "context": "The company was founded in 1990. It has 500 employees.",
        "expected_answer": "Not available",
        "should_NOT_contain": ["million", "$", "1B"],
    },
    {
        "name": "Conflicting Information",
        "question": "When was it founded?",
        "context": "Document 1: Founded in 1990. Document 2: Established in 1995.",
        "expected_answer": "conflict",  # Should mention contradiction
        "should_NOT_contain": None,  # Flexible
    },
    {
        "name": "Partial Information",
        "question": "What's the author's background?",
        "context": "John is a software engineer.",
        "expected_answer": "software engineer",
        "should_NOT_contain": ["PhD", "Harvard", "Nobel"],
    },
]


if __name__ == "__main__":
    print("Prompt Library Loaded Successfully")
    print(f"Available prompt versions: {list(PROMPT_VERSIONS.keys())}")
    print(f"Test cases for hallucination detection: {len(HALLUCINATION_TEST_CASES)}")