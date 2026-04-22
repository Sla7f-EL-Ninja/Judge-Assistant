"""
state.py

ChatReasonerState TypedDict, Pydantic schemas, and reducer helpers.
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Allowed tool names (validated by plan_validator)
# ---------------------------------------------------------------------------

ALLOWED_TOOLS = frozenset({"case_doc_rag", "civil_law_rag"})

# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------


_STEP_RESULTS_RESET = {"__reset__": True}


def _add_or_reset_step_results(a: List[dict], b: List[dict]) -> List[dict]:
    """Append reducer with reset support.

    An empty b list signals a full reset (used by replanner to wipe old results
    before a new execution wave). A non-empty b is appended to a as usual.
    Legacy sentinel check retained for backwards compatibility.
    """
    if not b:
        return []
    if b[0] == _STEP_RESULTS_RESET:
        return list(b[1:])
    return a + b


def _merge_step_failures(
    a: Dict[str, int], b: Dict[str, int]
) -> Dict[str, int]:
    """Merge two step-failure dicts, keeping the higher count per step_id.

    Used as a state reducer so parallel Send-dispatched step_workers can each
    contribute their failure increments without clobbering sibling counters.
    """
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0), v)
    return out


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ChatReasonerState(TypedDict):
    # --- Input ---
    original_query: str
    case_id: str
    conversation_history: List[dict]
    escalation_reason: str

    # --- Planning ---
    plan: List[dict]                       # list of PlanStep.model_dump()
    plan_validation_status: str            # "pending" | "valid" | "invalid"
    plan_validation_feedback: str
    validator_retry_count: int             # cap = 3

    # --- Execution (parallel Send reducers) ---
    step_results: Annotated[List[dict], _add_or_reset_step_results]
    step_failures: Annotated[Dict[str, int], _merge_step_failures]

    # --- Replan ---
    replan_count: int                      # cap = 2
    replan_trigger_step_id: Optional[str]
    replan_trigger_error: Optional[str]

    # --- Synthesis ---
    run_count: int                         # safety net cap = 2
    synthesis_attempts: int
    final_answer: str
    final_sources: List[str]

    # --- Control ---
    status: str                            # "running" | "succeeded" | "failed"
    error_message: Optional[str]
    synth_sufficient: bool                 # set by synthesizer_node for synth_router

    # --- Trace ---
    session_id: str                        # f"{case_id}::{iso_ts}"
    started_at: str
    tool_calls_log: Annotated[List[dict], add]
    replan_events: Annotated[List[dict], add]


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    step_id: str
    tool: str
    query: str
    depends_on: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    steps: List[PlanStep]
    parallel_groups_note: str = Field(
        description="Explicit justification of which steps run in parallel"
    )


class PlanValidationResult(BaseModel):
    valid: bool
    failed_checks: List[str]
    feedback: str


class StepResult(BaseModel):
    step_id: str
    tool: str
    query: str
    status: str                            # "success" | "failure" | "skipped"
    response: str
    sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    raw_output: Dict[str, Any] = Field(default_factory=dict)


class SynthesizerDecision(BaseModel):
    answer: str
    sufficient: bool
    insufficiency_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Step-worker payload TypedDict (received via LangGraph Send)
# ---------------------------------------------------------------------------


class StepWorkerPayload(TypedDict):
    step: dict
    case_id: str
    conversation_history: List[dict]
    prior_results: List[dict]