import streamlit as st
import requests
from components.sidebar import render_sidebar
from components.upload import render_upload_tab
from components.collections import render_collections_tab
from components.results import render_results, render_answer, render_debug

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Neural Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
options = render_sidebar()
active_collection = options["collection"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Neural Search")
st.caption("Hybrid semantic search — BM25 + Dense Retrieval + Reciprocal Rank Fusion")

if active_collection:
    st.markdown(f"Searching in: **{active_collection}**")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_upload, tab_collections, tab_history = st.tabs([
    "🔍 Search", "📂 Upload", "🗂 Collections", "🕓 History"
])

# ── Search tab ────────────────────────────────────────────────────────────────
with tab_search:
    if not active_collection:
        st.info("👈 Create a collection and upload documents to get started")
    else:
        query = st.text_input(
            "Ask a question or describe what you're looking for",
            placeholder="e.g. What are the payment terms for contract renewal?",
            label_visibility="collapsed",
        )

        col_btn, col_debug = st.columns([1, 6])
        search_clicked = col_btn.button("Search", type="primary", use_container_width=True)
        debug_mode = col_debug.toggle("Debug view (show BM25 vs Neural vs RRF breakdown)", value=False)

        if search_clicked and query.strip():
            with st.spinner("Searching..."):
                try:
                    if debug_mode:
                        resp = requests.get(
                            f"{API_BASE}/search/debug",
                            params={"query": query, "collection": active_collection, "k": options["top_k"]},
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            render_debug(resp.json())
                        else:
                            st.error(f"Debug failed: {resp.text}")
                    else:
                        resp = requests.post(
                            f"{API_BASE}/search",
                            json={
                                "query": query,
                                "collection": active_collection,
                                "k": options["top_k"],
                                "mode": options["mode"],
                                "synthesize": options["synthesize"],
                            },
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state.query_history.append({
                                "query": query,
                                "collection": active_collection,
                                "latency_ms": data["latency_ms"],
                                "mode": data["mode"],
                                "results": len(data["results"]),
                            })
                            if options["synthesize"] and data.get("synthesis"):
                                render_answer(data["synthesis"])
                            render_results(data["results"], data["latency_ms"], data["mode"])
                        else:
                            st.error(f"Search error: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach API — run `./run.sh api`")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        elif search_clicked:
            st.warning("Please enter a query")

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    render_upload_tab(active_collection)

# ── Collections tab ───────────────────────────────────────────────────────────
with tab_collections:
    render_collections_tab()

# ── History tab ───────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("🕓 Query History")
    history = st.session_state.query_history
    if not history:
        st.info("No queries yet")
    else:
        for entry in reversed(history[-30:]):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
                c1.markdown(f"**{entry['query']}**")
                c2.caption(f"📁 {entry['collection']}")
                c3.caption(f"`{entry['mode']}`")
                c4.caption(f"{entry['latency_ms']} ms")
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
