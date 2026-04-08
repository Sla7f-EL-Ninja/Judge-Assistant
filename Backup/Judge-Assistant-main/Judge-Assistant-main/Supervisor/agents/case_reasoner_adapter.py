"""
case_reasoner_adapter.py

Adapter for the Case Reasoner agent (Case Reasoner/case_reasoner.py).

Wraps the compiled ``app`` graph and returns an AgentResult with the
identified legal issues and judicial conclusion.
"""

import logging
import os
import sys
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class CaseReasonerAdapter(AgentAdapter):
    """Thin wrapper around the Case Reasoner LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Run judicial reasoning on the judge query and case summary.

        Parameters
        ----------
        query:
            The judge query or directive.
        context:
            Should contain ``case_summary`` (str).  May also contain
            ``agent_results.summarize`` if summarisation ran earlier in
            the same turn.
        """
        try:
            # Add Case Reasoner directory to path
            reasoner_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "Case Reasoner",
            )
            reasoner_dir = os.path.normpath(reasoner_dir)
            if reasoner_dir not in sys.path:
                sys.path.insert(0, reasoner_dir)

            from dotenv import load_dotenv
            load_dotenv()

            from case_reasoner import app

            # Try to obtain a case summary from context
            case_summary = context.get("case_summary", "")
            if not case_summary:
                # Fall back to summarisation output from an earlier agent
                summarize_result = (
                    context.get("agent_results") or {}
                ).get("summarize")
                if summarize_result and isinstance(summarize_result, dict):
                    case_summary = summarize_result.get(
                        "rendered_brief", ""
                    )

            initial_state = {
                "judge_query": query,
                "case_summary": case_summary,
                "identified_issues": [],
                "decomposed_elements": {},
                "law_retrievals": {},
                "case_retrievals": {},
                "factual_analysis": [],
                "legal_analysis": [],
                "conclusion": "",
                "intermediate_steps": [],
                "error_log": [],
            }

            result = app.invoke(initial_state)

            conclusion = result.get("conclusion", "")
            identified_issues = result.get("identified_issues", [])

            # Build a human-readable response
            parts = []
            if identified_issues:
                parts.append("المسائل القانونية المحددة:")
                for issue in identified_issues:
                    title = issue.get("issue_title", "")
                    domain = issue.get("legal_domain", "")
                    parts.append(f"- {title} ({domain})")
            if conclusion:
                parts.append(f"\nالخلاصة:\n{conclusion}")

            response = "\n".join(parts) if parts else ""

            return AgentResult(
                response=response,
                sources=[],
                raw_output={
                    "identified_issues": identified_issues,
                    "conclusion": conclusion,
                    "intermediate_steps": result.get("intermediate_steps", []),
                    "error_log": result.get("error_log", []),
                },
            )

        except Exception as exc:
            error_msg = f"Case Reasoner adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
