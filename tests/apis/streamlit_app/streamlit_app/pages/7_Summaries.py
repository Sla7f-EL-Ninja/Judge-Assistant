"""Case Summaries page — 3 tabs."""

import streamlit as st
from utils.display import show_response

st.title("Summaries")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_get, tab_generate, tab_brief = st.tabs(["Get Summary", "Generate", "Case Brief"])

# -- Get Summary --------------------------------------------------------------
with tab_get:
    st.subheader("Get Stored Summary")
    st.markdown("`GET /api/v1/cases/{case_id}/summary`")

    case_id = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="sum_get_case_id")

    if st.button("Fetch Summary"):
        if not case_id.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.get_summary(case_id.strip())
            show_response(status, body, elapsed)

            if status == 200 and "summary" in body:
                st.divider()
                st.subheader("Summary")
                st.markdown(body["summary"])
                if body.get("sources"):
                    st.subheader("Sources")
                    for src in body["sources"]:
                        st.markdown(f"- {src}")

# -- Generate -----------------------------------------------------------------
with tab_generate:
    st.subheader("Generate Summary")
    st.markdown("`POST /api/v1/cases/{case_id}/summary/generate`")
    st.caption("Runs the full summarization pipeline (Nodes 0–5). Overwrites any previous summary.")

    case_id_gen = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="sum_gen_case_id")

    if st.button("Generate Summary"):
        if not case_id_gen.strip():
            st.error("Case ID is required.")
        else:
            with st.spinner("Running summarization pipeline (may take 30–90s)..."):
                status, body, elapsed = client.generate_summary(case_id_gen.strip())
            show_response(status, body, elapsed)

            if status == 200:
                st.success(
                    f"Summary generated — {body.get('sources_count', 0)} sources. "
                    f"{body.get('message', '')}"
                )

# -- Case Brief ---------------------------------------------------------------
with tab_brief:
    st.subheader("Get Case Brief")
    st.markdown("`GET /api/v1/cases/{case_id}/case-brief`")
    st.caption("Returns the structured 7-section Arabic judicial brief.")

    case_id_brief = st.text_input("Case ID", value=st.session_state.get("last_case_id", ""), key="brief_case_id")

    if st.button("Fetch Case Brief"):
        if not case_id_brief.strip():
            st.error("Case ID is required.")
        else:
            status, body, elapsed = client.get_case_brief(case_id_brief.strip())
            show_response(status, body, elapsed)

            if status == 200 and body.get("case_brief"):
                st.divider()
                st.subheader("Case Brief Sections")
                brief = body["case_brief"]
                section_labels = {
                    "dispute_summary": "Dispute Summary / ملخص النزاع",
                    "uncontested_facts": "Uncontested Facts / الوقائع غير المتنازع عليها",
                    "key_disputes": "Key Disputes / النقاط الخلافية",
                    "party_requests": "Party Requests / طلبات الأطراف",
                    "party_defenses": "Party Defenses / دفوع الأطراف",
                    "submitted_documents": "Submitted Documents / المستندات المقدمة",
                    "legal_questions": "Legal Questions / المسائل القانونية",
                }
                for key, label in section_labels.items():
                    val = brief.get(key, "")
                    with st.expander(label, expanded=bool(val)):
                        if isinstance(val, list):
                            for item in val:
                                st.markdown(f"- {item}")
                        elif val:
                            st.markdown(str(val))
                        else:
                            st.caption("No content.")
