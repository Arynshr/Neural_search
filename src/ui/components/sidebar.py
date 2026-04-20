import streamlit as st
import requests
from neural_search.config import settings

API_BASE = "http://localhost:8000"


def render_sidebar() -> dict:
    """
    Renders sidebar controls and index stats.
    Returns a dict of user-selected options.
    """
    st.sidebar.title("⚙️ Neural Search")
    st.sidebar.markdown("---")

    # Index stats
    st.sidebar.subheader("Index Status")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code == 200:
            health = resp.json()
            col1, col2 = st.sidebar.columns(2)
            col1.metric("BM25 Chunks", health["bm25_chunks"])
            col2.metric("Qdrant Points", health["qdrant_points"])
            sync = health["index_in_sync"]
            st.sidebar.success("Indexes in sync ✓") if sync else st.sidebar.error("Index drift detected ✗")
        else:
            st.sidebar.warning("API unavailable")
    except Exception:
        st.sidebar.error("Cannot reach API — is it running?")

    st.sidebar.markdown("---")

    # Search options
    st.sidebar.subheader("Search Options")
    mode = st.sidebar.selectbox(
        "Retrieval Mode",
        options=["hybrid", "sparse", "dense"],
        index=0,
        help="hybrid = BM25 + Dense + RRF | sparse = BM25 only | dense = Qdrant only",
    )
    top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=20, value=5)
    synthesize = st.sidebar.toggle("Generate Answer (Groq)", value=False)

    st.sidebar.markdown("---")

    # Reset index
    st.sidebar.subheader("Danger Zone")
    if st.sidebar.button("🗑️ Reset Both Indexes", type="secondary"):
        try:
            r = requests.delete(f"{API_BASE}/index", timeout=10)
            if r.status_code == 200:
                st.sidebar.success("Indexes reset successfully")
                st.rerun()
            else:
                st.sidebar.error("Reset failed")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    return {"mode": mode, "top_k": top_k, "synthesize": synthesize}
