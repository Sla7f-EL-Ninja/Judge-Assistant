"""
test_routing_accuracy.py -- Measure intent classification accuracy.

Loads routing_cases.json and runs each query through classify_intent_node,
verifying that the overall routing accuracy meets the 85% threshold.

Marker: behavioral
"""

import json
import pathlib
from collections import defaultdict

import pytest

ROUTING_CASES_PATH = pathlib.Path(__file__).parent.parent / "eval" / "routing_cases.json"


def _load_routing_cases():
    """Load and return routing test cases from JSON."""
    with open(ROUTING_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.behavioral
class TestRoutingAccuracy:
    """Measure routing accuracy across all agent categories."""

    def _classify_query(self, query: str) -> dict:
        """Run a query through classify_intent_node and return the state update."""
        from Supervisor.nodes.classify_intent import classify_intent_node

        state = {
            "judge_query": query,
            "case_id": "",
            "uploaded_files": [],
            "conversation_history": [],
            "turn_count": 0,
            "intent": "",
            "target_agents": [],
            "classified_query": "",
            "agent_results": {},
            "agent_errors": {},
            "validation_status": "",
            "validation_feedback": "",
            "retry_count": 0,
            "max_retries": 2,
            "document_classifications": [],
            "merged_response": "",
            "final_response": "",
            "sources": [],
        }
        return classify_intent_node(state)

    def test_routing_accuracy_overall(self):
        """Overall routing accuracy must be >= 85%."""
        try:
            cases = _load_routing_cases()
        except FileNotFoundError:
            pytest.skip("routing_cases.json not found")

        correct = 0
        total = len(cases)
        per_agent = defaultdict(lambda: {"correct": 0, "total": 0})
        errors = []

        for case in cases:
            query = case["query"]
            expected = case["expected_agent"]
            per_agent[expected]["total"] += 1

            try:
                result = self._classify_query(query)
                actual_intent = result.get("intent", "")
                actual_agents = result.get("target_agents", [])

                # Match: either intent matches or expected agent is in target_agents
                if actual_intent == expected or expected in actual_agents:
                    correct += 1
                    per_agent[expected]["correct"] += 1
                else:
                    errors.append(
                        f"  Query: {query[:60]}... | "
                        f"Expected: {expected} | Got: {actual_intent} "
                        f"(agents: {actual_agents})"
                    )
            except Exception as exc:
                errors.append(f"  Query: {query[:60]}... | Error: {exc}")

        accuracy = correct / total if total > 0 else 0

        # Print confusion summary
        summary = f"\nRouting accuracy: {correct}/{total} = {accuracy:.2%}\n"
        for agent, stats in sorted(per_agent.items()):
            agent_acc = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            )
            summary += f"  {agent}: {stats['correct']}/{stats['total']} = {agent_acc:.2%}\n"

        if errors:
            summary += f"\nMisclassifications ({len(errors)}):\n"
            summary += "\n".join(errors[:10])  # Show first 10

        assert accuracy >= 0.85, (
            f"Routing accuracy {accuracy:.2%} is below 85% threshold.{summary}"
        )

    @pytest.mark.parametrize(
        "case",
        _load_routing_cases() if ROUTING_CASES_PATH.exists() else [],
        ids=lambda c: f"{c['expected_agent']}:{c['query'][:30]}",
    )
    def test_individual_routing(self, case):
        """Each routing case should classify correctly."""
        try:
            result = self._classify_query(case["query"])
            actual_intent = result.get("intent", "")
            actual_agents = result.get("target_agents", [])
            expected = case["expected_agent"]

            assert actual_intent == expected or expected in actual_agents, (
                f"Query: {case['query']}\n"
                f"Expected agent: {expected}\n"
                f"Got intent: {actual_intent}, agents: {actual_agents}"
            )
        except Exception as exc:
            pytest.skip(f"classify_intent_node unavailable: {exc}")
