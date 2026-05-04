"""Legal corpus search page."""

import streamlit as st
from utils.display import show_response

st.title("Legal Search")

client = st.session_state.get("client")
if not client:
    st.error("Configure API connection in the sidebar first.")
    st.stop()

tab_search, tab_lookup = st.tabs(["Free-Text Search", "Article Lookup"])

# -- Tab 1: RAG free-text search ------------------------------------------
with tab_search:
    st.markdown("`GET /api/v1/legal/search`")
    st.caption("Direct search of the civil law corpus — no supervisor graph invoked.")

    query = st.text_input("Query", placeholder="Enter your question in Arabic...", key="rag_query")
    corpus_s = st.selectbox("Corpus", ["civil"], key="rag_corpus")

    if st.button("Search", key="btn_search"):
        if not query.strip():
            st.error("Query is required.")
        else:
            with st.spinner("Searching legal corpus..."):
                status, body, elapsed = client.legal_search(query.strip(), corpus_s)
            show_response(status, body, elapsed)

            if status == 200:
                st.divider()
                col1, col2 = st.columns(2)
                confidence = body.get("retrieval_confidence")
                if confidence is not None:
                    col1.metric("Retrieval Confidence", f"{confidence:.2f}")
                from_cache = body.get("from_cache", False)
                col2.markdown(
                    f"**Cache:** :{'green' if from_cache else 'gray'}[{'HIT' if from_cache else 'MISS'}]"
                )
                if body.get("answer"):
                    st.subheader("Answer")
                    st.markdown(body["answer"])
                if body.get("sources"):
                    st.subheader("Sources")
                    for src in body["sources"]:
                        st.markdown(f"- {src}")

# -- Tab 2: Structured article lookup -------------------------------------
with tab_lookup:
    st.markdown("`GET /api/v1/legal/article`")
    st.caption("Fetch articles directly by number, chapter, or section — no LLM.")

    corpus_l = st.selectbox("Corpus", ["civil"], key="lookup_corpus")
    col1, col2, col3 = st.columns(3)
    article_no = col1.number_input("Article No.", min_value=1, step=1, value=None, placeholder="e.g. 190")
    chapter    = col2.text_input("Chapter", placeholder="optional")
    section    = col3.text_input("Section", placeholder="optional")

    if st.button("Lookup", key="btn_lookup"):
        if not any([article_no, chapter.strip(), section.strip()]):
            st.error("At least one filter is required.")
        else:
            with st.spinner("Fetching articles..."):
                status, body, elapsed = client.legal_article_lookup(
                    corpus=corpus_l,
                    article_no=int(article_no) if article_no else None,
                    chapter=chapter.strip() or None,
                    section=section.strip() or None,
                )
            show_response(status, body, elapsed)

            if status == 200:
                st.divider()
                st.caption(f"Found **{body.get('count', 0)}** article(s)")
                for art in body.get("articles", []):
                    label = f"المادة {art['index']}" if art.get("index") else "Article"
                    with st.expander(label):
                        if art.get("chapter"):
                            st.caption(f"Chapter: {art['chapter']}")
                        if art.get("section"):
                            st.caption(f"Section: {art['section']}")
                        st.markdown(art.get("text", ""))