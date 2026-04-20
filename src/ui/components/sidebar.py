import streamlit as st
import requests

API_BASE = "http://localhost:8000"


def render_sidebar() -> dict:
    st.sidebar.title("🔍 Neural Search")
    st.sidebar.caption("Hybrid semantic search")
    st.sidebar.markdown("---")

    # Load collections
    try:
        resp = requests.get(f"{API_BASE}/collections", timeout=3)
        collections = resp.json() if resp.status_code == 200 else []
    except Exception:
        st.sidebar.error("⚠️ API unreachable")
        return {"collection": None, "mode": "hybrid", "top_k": 5, "synthesize": False}

    # Collection switcher
    st.sidebar.subheader("🗂 Collections")
    if not collections:
        st.sidebar.info("No collections yet — create one in the Collections tab")
        active_slug = None
    else:
        col_options = {c["name"]: c["slug"] for c in collections}
        col_labels = list(col_options.keys())

        # Restore last used collection
        default_idx = 0
        if "active_collection" in st.session_state:
            saved = st.session_state.active_collection
            matching = [i for i, s in enumerate(col_options.values()) if s == saved]
            if matching:
                default_idx = matching[0]

        selected_name = st.sidebar.radio(
            "Active collection",
            col_labels,
            index=default_idx,
            label_visibility="collapsed",
        )
        active_slug = col_options[selected_name]
        st.session_state.active_collection = active_slug

        # Stats for active collection
        active = next(c for c in collections if c["slug"] == active_slug)
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Files", len(active["files"]))
        c2.metric("Chunks", active["total_chunks"])
        st.sidebar.caption(f"Updated: {active['updated_at'][:10]}")

    st.sidebar.markdown("---")

    # Search options
    st.sidebar.subheader("⚙️ Search Options")
    mode = st.sidebar.selectbox(
        "Retrieval Mode",
        ["hybrid", "sparse", "dense"],
        help="hybrid = BM25 + Neural + RRF",
    )
    top_k = st.sidebar.slider("Top K Results", 1, 20, 5)
    synthesize = st.sidebar.toggle("🤖 Generate Answer (Groq)", value=False)

    return {
        "collection": active_slug,
        "mode": mode,
        "top_k": top_k,
        "synthesize": synthesize,
    }
