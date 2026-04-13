"""case_doc_rag.models -- Pydantic models for structured LLM output.

Pure Pydantic only. No LangChain imports. No infrastructure.
"""

from typing import Optional

from pydantic import BaseModel, Field


class GradeQuestion(BaseModel):
    """Binary classifier for whether the judge's question is in scope."""

    score: str = Field(
        description=(
            "Classifier result. Respond strictly with 'Yes' if the judge's question "
            "is related in ANY way to Egyptian civil-case matters, including: "
            "civil procedure, substantive civil/commercial law, evidence, case documents, "
            "procedural history, case analysis, or ANY question that references or "
            "affects the case -- even indirectly. Err toward 'Yes' -- indirect or "
            "partial relevance still counts as in-scope. "
            "Respond 'No' only if the question is about criminal law, unrelated "
            "personal questions, or completely unconnected to civil law or the case."
        )
    )


class DocSelection(BaseModel):
    """Document selection classification result."""

    mode: str = Field(
        description=(
            "Classify the query into exactly one of these three modes:\n"
            "'retrieve_specific_doc' -- The judge is asking to GET or DISPLAY a "
            "document itself. Examples: 'هاتلي مذكرة المدعى عليه', "
            "'اعرض تقرير الخبير', 'فين صحيفة الاستئناف؟'\n"
            "'restrict_to_doc' -- The judge asks for INFORMATION FROM a specific "
            "document but not to return the document itself. Examples: "
            "'ايه أهم النقاط الواردة في مذكرة المدعى؟', "
            "'استخرج لي الوقائع الواردة في صحيفة الدعوى'\n"
            "'no_doc_specified' -- The judge does not refer to any specific document. "
            "Examples: 'ما هي الدفوع الشكلية المتاحة؟', 'لخص لي النزاع'"
        )
    )
    doc_id: Optional[str] = Field(
        default=None,
        description=(
            "The EXACT title string from the available documents list provided "
            "in the system prompt, or null/None if no document is referenced. "
            "You MUST NOT invent, guess, or paraphrase document names. "
            "Return the exact title as it appears in the list."
        ),
    )


class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    score: str = Field(
        description=(
            "A binary score ('Yes' or 'No') indicating whether the document "
            "contains specific legal facts, keywords, names, dates, procedural "
            "history, or legal references that could help answer the judge's query. "
            "The documents are Arabic-language Egyptian civil-case files. "
            "Answer 'Yes' if the chunk provides any useful context for answering "
            "the question; answer 'No' only if the chunk is completely unrelated."
        )
    )
