"""
case_reasoner.py

This module implements the core judicial reasoning agent using LangGraph.
It orchestrates the flow from issue extraction to legal analysis and final conclusion.

Project: AI Judge Assistant
Incremental Step: Foundation, State Definition, and Issue Extraction.
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import get_llm

# Load environment variables (ensure GOOGLE_API_KEY is set)
load_dotenv()

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------

class CaseReasonerState(TypedDict):
    """
    Represents the state of the judicial reasoning process.
    Maintains context across all nodes in the LangGraph workflow.
    """
    judge_query: str                  # The specific question or directive from the judge
    case_summary: str                 # Summarized facts and context of the case
    identified_issues: List[Dict[str, Any]] # Extracted legal issues: [{id, title, domain}]
    decomposed_elements: Dict[str, Any] # Breakdown of issues into factual/legal components
    law_retrievals: Dict[str, Any]    # Relevant laws retrieved for each issue
    case_retrievals: Dict[str, Any]   # Relevant precedents/documents retrieved
    factual_analysis: List[str]       # Findings of fact for each issue
    legal_analysis: List[str]         # Application of law to the facts
    conclusion: str                   # Final judicial reasoning and verdict
    intermediate_steps: List[str]     # Log of process steps for transparency
    error_log: List[str]              # Any errors or warnings encountered


# ---------------------------------------------------------
# 2. Tool Placeholders
# ---------------------------------------------------------

def civil_law_rag_tool(query: str) -> str:
    """
    Placeholder for the tool that retrieves relevant articles from Civil Law.
    """
    # TODO: Integrate with Civil-Law-RAG system
    return f"Stub: Retrieved law for {query}"

def case_documents_rag_tool(query: str) -> str:
    """
    Placeholder for the tool that retrieves relevant case documents or precedents.
    """
    # TODO: Integrate with Case Documents RAG system
    return f"Stub: Retrieved documents for {query}"


# ---------------------------------------------------------
# 3. Node Implementation: Issue Extraction
# ---------------------------------------------------------

# Structured Output Schema
class LegalIssue(BaseModel):
    issue_id: int = Field(description="Unique identifier for the issue")
    issue_title: str = Field(description="Brief title of the legal issue")
    legal_domain: str = Field(description="The area of law this issue pertains to (e.g., Tort, Contract, Property)")

class ExtractedIssues(BaseModel):
    issues: List[LegalIssue] = Field(description="List of identified legal issues")

def extract_issues_node(state: CaseReasonerState) -> Dict[str, Any]:
    """
    Identifies and extracts high-level legal issues from the judge's query and case summary.
    
    Judicial Logic:
    - Analyzes the conflict points between parties.
    - Identifies the core legal questions that must be answered to resolve the case.
    - Categorizes issues by legal domain to guide subsequent law retrieval.
    """
    
    # Initialize LLM via centralized tier system
    llm = get_llm("high")
    structured_llm = llm.with_structured_output(ExtractedIssues)

    # Prompt Template
    prompt = f"""
    أنت كاتب قضائي أول تُعاون أحد القضاة.
    مهمتك هي استخراج المسائل القانونية الجوهرية من طلب القاضي وملخص الدعوى المقدمين أدناه.

    طلب القاضي:
    {state['judge_query']}

    ملخص الدعوى:
    {state['case_summary']}

    التعليمات:
    1. حدّد المسائل القانونية المستقلة التي يتعين الفصل فيها.
    2. لكل مسألة، قدم عنوانًا موجزًا وحدد المجال القانوني المرتبط بها.
    3. التزم بالموضوعية والدقة ولا تقم بتطبيق النصوص القانونية في هذه المرحلة.
    4. ركّز فقط على المسائل القانونية العامة ذات المستوى العالي.

    قم بإرجاع المسائل في صيغة منظمة وواضحة.
    """

    try:
        # Invoke LLM
        result = structured_llm.invoke(prompt)
        
        # Format the result for the state
        formatted_issues = [
            {
                "issue_id": issue.issue_id,
                "issue_title": issue.issue_title,
                "legal_domain": issue.legal_domain
            }
            for issue in result.issues
        ]
        
        return {
            "identified_issues": formatted_issues,
            "intermediate_steps": state.get("intermediate_steps", []) + ["Extracted legal issues."]
        }
    except Exception as e:
        error_msg = f"Error in extract_issues_node: {str(e)}"
        return {
            "error_log": state.get("error_log", []) + [error_msg],
            "intermediate_steps": state.get("intermediate_steps", []) + ["Failed to extract issues."]
        }


# ---------------------------------------------------------
# 4. LangGraph Configuration
# ---------------------------------------------------------

# Initialize Workflow
workflow = StateGraph(CaseReasonerState)

# Add Nodes
workflow.add_node("extract_issues", extract_issues_node)

# Set Entry Point
workflow.set_entry_point("extract_issues")

# Define Edges (Start -> extract_issues -> End)
workflow.add_edge("extract_issues", END)

# Compile Graph
app = workflow.compile()

# ---------------------------------------------------------
# Optional: Entry Point for Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example input for testing
    test_state = {
        "judge_query": "Determine the liability of the defendant in the car accident case.",
        "case_summary": "On Jan 1st, Defendant's car hit Plaintiff's car at an intersection. Plaintiff claims Defendant ran a red light.",
        "identified_issues": [],
        "decomposed_elements": {},
        "law_retrievals": {},
        "case_retrievals": {},
        "factual_analysis": [],
        "legal_analysis": [],
        "conclusion": "",
        "intermediate_steps": [],
        "error_log": []
    }
    
    # Run the graph
    # result = app.invoke(test_state)
    # print(json.dumps(result["identified_issues"], indent=2, ensure_ascii=False))
    print("Case Reasoner workflow compiled successfully.")
