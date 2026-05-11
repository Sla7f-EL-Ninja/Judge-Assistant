"""Case Reasoning page."""

import streamlit as st
from utils.display import show_response

st.title("Case Reasoning")
st.markdown("`GET /api/v1/cases/{case_id}/case-reasoning`")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

case_id = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""))

if st.button("Fetch Case Reasoning"):
    if not case_id.strip():
        st.error("Case ID is required.")
    else:
        status, body, elapsed = client.get_case_reasoning(case_id.strip())
        show_response(status, body, elapsed)

        if status == 200:
            st.divider()

            # Final Report
            if body.get("final_report"):
                st.subheader("Final Report")
                st.markdown(body["final_report"])

            # Case-level confidence
            if body.get("case_level_confidence"):
                st.subheader("Case-Level Confidence")
                st.json(body["case_level_confidence"])

            # Per-issue confidence table
            if body.get("per_issue_confidence"):
                st.subheader("Per-Issue Confidence")
                rows = body["per_issue_confidence"]
                st.dataframe(rows, use_container_width=True)

            # Issue analyses summary
            analyses = body.get("issue_analyses", [])
            if analyses:
                st.subheader(f"Issue Analyses ({len(analyses)} issues)")
                for i, analysis in enumerate(analyses):
                    label = analysis.get("issue", f"Issue {i + 1}")
                    with st.expander(label):
                        st.json(analysis)

            # Cross-issue relationships
            relationships = body.get("cross_issue_relationships", [])
            if relationships:
                with st.expander(f"Cross-Issue Relationships ({len(relationships)})"):
                    st.json(relationships)

            # Consistency conflicts
            conflicts = body.get("consistency_conflicts", [])
            if conflicts:
                with st.expander(f"Consistency Conflicts ({len(conflicts)})", expanded=bool(conflicts)):
                    st.json(conflicts)

            # Reconciliation paragraphs
            paragraphs = body.get("reconciliation_paragraphs", [])
            if paragraphs:
                with st.expander(f"Reconciliation Paragraphs ({len(paragraphs)})"):
                    for p in paragraphs:
                        st.markdown(p)
