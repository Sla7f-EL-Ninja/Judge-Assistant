"""
fallback.py
-----------
Terminal fallback nodes: off_topic and cannot_answer.
"""

from langsmith import traceable


@traceable(name="Off-Topic Node")
def off_topic_node(state: dict) -> dict:
    state["final_answer"] = (
        "يبدو أن السؤال المطروح خارج نطاق اختصاص هذا النظام. "
        "يرجى طرح سؤال متعلق بالقانون أو الأحكام القانونية ذات الصلة."
    )
    return state


@traceable(name="Cannot Answer Node")
def cannot_answer_node(state: dict) -> dict:
    reason = state.get("failure_reason", "تعذر العثور على مواد قانونية مناسبة.")
    state["final_answer"] = (
        "تعذر تقديم إجابة قانونية دقيقة على السؤال المطروح.\n\n"
        f"السبب:\n{reason}\n\n"
        "يرجى إعادة صياغة السؤال أو توضيح الوقائع بشكل أدق."
    )
    return state
