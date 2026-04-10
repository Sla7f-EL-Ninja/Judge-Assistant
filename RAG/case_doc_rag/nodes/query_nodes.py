"""case_doc_rag.nodes.query_nodes -- Query processing and fan-out nodes.

Nodes: questionRewriter, questionClassifier, offTopicResponse, dispatchQuestions
"""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send

from RAG.case_doc_rag.infrastructure import get_llm
from RAG.case_doc_rag.models import GradeQuestion
from RAG.case_doc_rag.prompts import QUESTION_CLASSIFIER_PROMPT, QUESTION_REWRITER_PROMPT
from RAG.case_doc_rag.state import AgentState, SubQuestionState

logger = logging.getLogger("case_doc_rag.query_nodes")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_json_list(raw: str) -> List[str]:
    """Parse a raw LLM response string into a List[str].

    Handles clean JSON, markdown-fenced JSON, embedded JSON arrays,
    and falls back to wrapping the raw string in a single-item list.
    """
    stripped = raw.strip()
    if not stripped:
        return []

    # 1. Clean JSON
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            result = [str(item).strip() for item in parsed if isinstance(item, str)]
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Markdown-fenced JSON
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if isinstance(parsed, list):
                result = [str(item).strip() for item in parsed if isinstance(item, str)]
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Embedded JSON array
    bracket_match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                result = [str(item).strip() for item in parsed if isinstance(item, str)]
                if result:
                    return result
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Fallback: single-item list
    return [stripped]


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def questionRewriter(state: AgentState) -> Dict[str, Any]:
    """Decompose the judge's query into standalone retrieval questions.

    Fixes: Bug 1 (query as HumanMessage), Bug 2 (first-turn skip),
    Bug 3 (JSON never parsed), Bug 7 (.format() anti-pattern).
    Always runs -- no conditional gate on conversation length.
    """
    request_id = state.get("request_id", "")
    query = state.get("query", "")

    if not query or not query.strip():
        logger.warning("[%s] Empty query received in questionRewriter", request_id)
        return {"error": "Empty query received", "sub_questions": []}

    # Build the LLM message list
    messages = [SystemMessage(content=QUESTION_REWRITER_PROMPT)]

    # Add user turns from conversation history for context
    for turn in state.get("conversation_history", []):
        if turn.get("role") == "user" and turn.get("content"):
            messages.append(HumanMessage(content=turn["content"]))

    # Current query
    messages.append(HumanMessage(content=query))

    try:
        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | get_llm("high")
        response = chain.invoke({})
        raw_content = response.content.strip()
    except Exception:
        logger.exception("[%s] LLM error in questionRewriter", request_id)
        # Fallback: use original query as single sub-question
        raw_content = None

    if raw_content:
        parsed_list = _extract_json_list(raw_content)
    else:
        parsed_list = []

    if not parsed_list:
        parsed_list = [query]

    logger.info(
        "[%s] questionRewriter: query=%r sub_questions=%s",
        request_id, query[:100], parsed_list,
    )
    return {"sub_questions": parsed_list}


def questionClassifier(state: AgentState) -> Dict[str, Any]:
    """Classify whether the rewritten question is on-topic.

    Fixes: Bug 4 (classifier evaluated raw query instead of rewritten question).
    Classifies sub_questions[0], NOT state["query"].
    """
    request_id = state.get("request_id", "")
    sub_questions = state.get("sub_questions", [])

    if not sub_questions:
        logger.warning("[%s] No sub_questions for questionClassifier", request_id)
        return {"on_topic": False}

    question_to_classify = sub_questions[0]

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUESTION_CLASSIFIER_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt | get_llm("high").with_structured_output(GradeQuestion)
        result = chain.invoke({"question": question_to_classify})
        on_topic = result.score.strip().lower() == "yes"
    except Exception:
        logger.exception("[%s] LLM error in questionClassifier", request_id)
        # Fail open -- transient API error should not refuse a valid question
        on_topic = True

    logger.info(
        "[%s] questionClassifier: question=%r on_topic=%s",
        request_id, question_to_classify[:100], on_topic,
    )
    return {"on_topic": on_topic}


def offTopicResponse(state: AgentState) -> Dict[str, Any]:
    """Return a standard Arabic refusal message for off-topic queries."""
    return {
        "final_answer": (
            "عذراً، لا يمكنني الإجابة على هذا السؤال لأنه خارج نطاق المستندات "
            "أو غير متعلق بالمسائل القانونية المرتبطة بالدعوى المدنية محل البحث."
        )
    }


def dispatchQuestions(state: AgentState) -> List[Send]:
    """Fan-out node: dispatch each sub-question to a parallel branch.

    Returns a list of Send objects (LangGraph fan-out API).
    Each Send targets 'retrieve_branch' with a complete SubQuestionState.
    """
    request_id = state.get("request_id", "")
    sub_questions = state.get("sub_questions", [])

    if not sub_questions:
        logger.error("[%s] No sub-questions to dispatch", request_id)
        # Return a Send with an error state rather than a dict
        # because this node must always return Send objects
        return [Send("retrieve_branch", {
            "sub_question": "",
            "case_id": state.get("case_id", ""),
            "conversation_history": state.get("conversation_history", []),
            "selected_doc_id": state.get("selected_doc_id"),
            "doc_selection_mode": state.get("doc_selection_mode", "no_doc_specified"),
            "request_id": request_id,
            "retrieved_docs": [],
            "proceedToGenerate": False,
            "rephraseCount": 0,
            "sub_answer": "",
            "sources": [],
            "found": False,
            "sub_answers": [],
        })]

    # Deduplicate sub-questions while preserving order
    seen: set = set()
    unique_questions = []
    for q in sub_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    sends = []
    for sub_q in unique_questions:
        sub_state = {
            "sub_question": sub_q,
            "case_id": state.get("case_id", ""),
            "conversation_history": state.get("conversation_history", []),
            "selected_doc_id": state.get("selected_doc_id"),
            "doc_selection_mode": state.get("doc_selection_mode", "no_doc_specified"),
            "request_id": request_id,
            "retrieved_docs": [],
            "proceedToGenerate": False,
            "rephraseCount": 0,
            "sub_answer": "",
            "sources": [],
            "found": False,
            "sub_answers": [],
        }
        sends.append(Send("retrieve_branch", sub_state))

    logger.info(
        "[%s] dispatchQuestions: dispatching %d branches",
        request_id, len(sends),
    )
    return sends
