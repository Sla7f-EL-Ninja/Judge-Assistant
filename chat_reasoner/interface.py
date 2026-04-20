"""
interface.py

Local copies of AgentAdapter and AgentResult for the chat_reasoner package.

These mirror the contract in Supervisor/agents/base.py but are defined here
so the package has zero imports from Supervisor/. The Supervisor dispatcher
(dispatch_agents.py) duck-types adapter.invoke() and reads result.response /
.sources / .raw_output / .error — identical field names ensure compatibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    response: str = Field(description="Main textual output from the agent")
    sources: List[str] = Field(default_factory=list)
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentAdapter(ABC):
    @abstractmethod
    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult: ...
