"""Case Reasoner Adapter — wraps the Case Reasoner LangGraph pipeline for the Supervisor."""
import logging
import os
import sys
from typing import Any, Dict, List

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)

_REASONER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Case Reasoner")
)


def _ensure_path() -> None:
    if _REASONER_DIR not in sys.path:
        sys.path.insert(0, _REASONER_DIR)


def _extract_sources(issue_analyses: List[Dict]) -> List[str]:
    """Collect unique article citations from all issue analyses."""
    seen: set = set()
    sources: List[str] = []
    for analysis in issue_analyses:
        for el in analysis.get("applied_elements") or []:
            for article_num in el.get("cited_articles") or []:
                key = f"م {article_num}"
                if key not in seen:
                    seen.add(key)
                    sources.append(key)
    return sources


class CaseReasonerAdapter(AgentAdapter):

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            _ensure_path()
            from dotenv import load_dotenv
            load_dotenv()
            from graph import build_case_reasoner_graph

            # Extract case_brief from prior summarize agent output
            summarize_result = (context.get("agent_results") or {}).get("summarize") or {}
            raw_output = summarize_result.get("raw_output") or {}
            case_brief = raw_output.get("case_brief") or {}
            rendered_brief = raw_output.get("rendered_brief") or ""

            # Normalize CaseBrief: may be a Pydantic model or plain dict
            if hasattr(case_brief, "dict"):
                case_brief = case_brief.dict()
            elif hasattr(case_brief, "model_dump"):
                case_brief = case_brief.model_dump()

            case_id: str = context.get("case_id") or ""

            initial_state = {
                "case_id": case_id,
                "judge_query": query,
                "case_brief": case_brief,
                "rendered_brief": rendered_brief,
                "identified_issues": [],
                "issue_analyses": [],
                "cross_issue_relationships": [],
                "consistency_conflicts": [],
                "reconciliation_paragraphs": [],
                "per_issue_confidence": [],
                "case_level_confidence": {},
                "final_report": "",
                "intermediate_steps": [],
                "error_log": [],
            }

            app = build_case_reasoner_graph()
            result = app.invoke(initial_state)

            final_report: str = result.get("final_report") or ""
            issue_analyses: List[Dict] = result.get("issue_analyses") or []
            sources = _extract_sources(issue_analyses)

            return AgentResult(
                response=final_report,
                sources=sources,
                raw_output={
                    "final_report": final_report,
                    "identified_issues": result.get("identified_issues") or [],
                    "issue_analyses": issue_analyses,
                    "case_level_confidence": result.get("case_level_confidence") or {},
                    "per_issue_confidence": result.get("per_issue_confidence") or [],
                    "intermediate_steps": result.get("intermediate_steps") or [],
                    "error_log": result.get("error_log") or [],
                },
            )

        except Exception as exc:
            error_msg = f"Case Reasoner adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
