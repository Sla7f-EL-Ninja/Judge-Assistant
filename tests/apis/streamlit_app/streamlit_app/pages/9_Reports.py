"""Async Report generation page — 3 tabs."""

import streamlit as st
from utils.display import show_response

st.title("Reports")
st.caption("Async pipeline: Summarizer → Case Reasoner. Poll for completion.")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_generate, tab_poll, tab_list = st.tabs(["Generate", "Poll Status", "List Jobs"])

# -- Generate -----------------------------------------------------------------
with tab_generate:
    st.subheader("Generate Report")
    st.markdown("`POST /api/v1/cases/{case_id}/reports/generate`")
    st.caption("Kicks off the async pipeline and returns a job_id immediately (HTTP 202).")

    case_id_gen = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="rep_gen_case_id")

    if st.button("Generate Report"):
        if not case_id_gen.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.generate_report(case_id_gen.strip())
            show_response(status, body, elapsed)

            if status == 202 and body.get("job_id"):
                st.session_state["last_report_id"] = body["job_id"]
                st.success(f"Job ID stored: {body['job_id']} — use Poll Status tab to check progress.")

# -- Poll Status --------------------------------------------------------------
with tab_poll:
    st.subheader("Poll Report Status")
    st.markdown("`GET /api/v1/cases/{case_id}/reports/{report_id}`")

    case_id_poll = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="rep_poll_case_id")
    report_id = st.text_input("Report ID", value=st.session_state.get("last_report_id", ""), key="rep_poll_id")

    if st.button("Check Status"):
        if not case_id_poll.strip() or not report_id.strip():
            st.error("Case ID and Report ID are required.")
        else:
            status, body, elapsed = client.get_report(case_id_poll.strip(), report_id.strip())
            show_response(status, body, elapsed)

            if status == 200:
                job_status = body.get("status", "unknown")
                color_map = {
                    "queued": "blue",
                    "running": "orange",
                    "completed": "green",
                    "failed": "red",
                }
                st.markdown(
                    f"**Job Status:** :{color_map.get(job_status, 'gray')}[{job_status.upper()}]"
                )

                if job_status == "completed":
                    st.divider()

                    if body.get("summary"):
                        with st.expander("Summary", expanded=True):
                            summary = body["summary"]
                            st.markdown(summary.get("summary", ""))
                            if summary.get("sources"):
                                st.subheader("Sources")
                                for src in summary["sources"]:
                                    st.markdown(f"- {src}")

                    if body.get("case_reasoning"):
                        with st.expander("Case Reasoning — Final Report", expanded=True):
                            cr = body["case_reasoning"]
                            st.markdown(cr.get("final_report", ""))
                            if cr.get("case_level_confidence"):
                                st.json(cr["case_level_confidence"])

                elif job_status == "failed":
                    st.error(f"Pipeline failed: {body.get('error', 'Unknown error')}")

# -- List Jobs ----------------------------------------------------------------
with tab_list:
    st.subheader("List Report Jobs")
    st.markdown("`GET /api/v1/cases/{case_id}/reports`")

    case_id_list = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="rep_list_case_id")
    col1, col2 = st.columns(2)
    skip = col1.number_input("Skip", min_value=0, value=0, step=1, key="rep_skip")
    limit = col2.number_input("Limit", min_value=1, max_value=100, value=20, step=1, key="rep_limit")

    if st.button("List Reports"):
        if not case_id_list.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.list_reports(
                case_id_list.strip(), skip=int(skip), limit=int(limit)
            )
            show_response(status, body, elapsed)

            if status == 200 and body.get("jobs"):
                st.divider()
                st.subheader(f"Jobs ({body.get('total', 0)} total)")
                rows = [
                    {
                        "report_id": j.get("report_id", ""),
                        "status": j.get("status", ""),
                        "created_at": j.get("created_at", ""),
                        "completed_at": j.get("completed_at", ""),
                    }
                    for j in body["jobs"]
                ]
                st.dataframe(rows, use_container_width=True)
