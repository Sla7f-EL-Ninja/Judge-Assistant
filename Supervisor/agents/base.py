"""
base.py

Abstract base class for agent adapters and the AgentResult model.

Every worker agent in the system gets a thin adapter that implements this
interface so the supervisor can invoke them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Standardised result returned by every agent adapter."""

    response: str = Field(
        description="The main textual output from the agent"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Citations or references produced by the agent",
    )
    raw_output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full agent state / output for validation purposes",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the agent invocation failed",
    )


class AgentAdapter(ABC):
    """Uniform interface that every worker-agent adapter must implement."""

    @abstractmethod
    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Run the wrapped worker agent and return a standardised result.

        Parameters
        ----------
        query:
            The (possibly rewritten) judge query.
        context:
            Additional data the agent might need, drawn from SupervisorState.
            Typical keys: ``uploaded_files``, ``case_id``,
            ``conversation_history``, ``agent_results`` (from earlier agents
            in a multi-agent turn).

        Returns
        -------
        AgentResult
        """
        ...
