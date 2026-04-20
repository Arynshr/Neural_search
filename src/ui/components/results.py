import streamlit as st

SOURCE_BADGE = {
    "sparse": " BM25",
    "dense": " Neural",
    "dense+sparse": " Both",
    "sparse+dense": " Both",
}


def render_answer(synthesis: dict):
    """Renders the Groq synthesized answer card."""
    st.markdown("### 💡 Answer")
    st.info(synthesis["answer"])
    with st.expander("Sources used"):
        for s in synthesis.get("sources_used", []):
            st.markdown(f"- **{s['source_file']}** — Page {s['page']}")
    st.caption(f"Model: `{synthesis['model']}`")
    st.markdown("---")


def render_results(results: list[dict], latency_ms: float, mode: str):
    """Renders ranked result cards with metadata and score breakdown."""
    st.markdown(
        f"**{len(results)} results** &nbsp;|&nbsp; "
        f"mode: `{mode}` &nbsp;|&nbsp; "
        f"latency: `{latency_ms} ms`",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for result in results:
        badge = SOURCE_BADGE.get(result.get("source", ""), "⚪ Unknown")
        score_key = "rrf_score" if "rrf_score" in result else "score"
        score_val = round(result.get(score_key, 0.0), 4)

        with st.expander(
            f"#{result['rank']}  {badge}  —  **{result['source_file']}**  "
            f"p.{result['page']}  &nbsp; score: `{score_val}`",
            expanded=result["rank"] <= 3,
        ):
            st.markdown(result["text"])
            col1, col2, col3 = st.columns(3)
            col1.caption(f"📄 {result['source_file']}")
            col2.caption(f"📃 Page {result['page']}")
            col3.caption(f"🔤 {result.get('token_count', '—')} tokens")


def render_debug(debug: dict):
    """Renders raw debug output — BM25, Dense, and RRF side by side."""
    st.markdown("### 🔬 Debug View")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**BM25 (Sparse)**")
        for r in debug.get("sparse", []):
            st.caption(f"#{r['rank']} score={round(r['score'], 4)}")
            st.markdown(f"> {r['text'][:120]}...")

    with col2:
        st.markdown("**Dense (Qdrant)**")
        for r in debug.get("dense", []):
            st.caption(f"#{r['rank']} score={round(r['score'], 4)}")
            st.markdown(f"> {r['text'][:120]}...")

    with col3:
        st.markdown("**Hybrid RRF**")
        for r in debug.get("hybrid_rrf", []):
            st.caption(f"#{r['rank']} rrf={round(r.get('rrf_score', 0), 6)}  [{r['source']}]")
            st.markdown(f"> {r['text'][:120]}...")
