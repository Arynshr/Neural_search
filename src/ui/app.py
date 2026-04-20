import requests
import streamlit as st
from components.sidebar import render_sidebar
from components.upload import render_upload
from components.results import render_results, render_answer, render_debug

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Neural Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []   # list of {query, latency_ms, mode}

# ── Sidebar ───────────────────────────────────────────────────────────────────
options = render_sidebar()

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🔍 Neural Search")
st.caption("Hybrid semantic search — BM25 + Dense Retrieval + RRF")
st.markdown("---")

tab_search, tab_upload, tab_history = st.tabs(["Search", "Upload Documents", "Query History"])

# ── Search tab ────────────────────────────────────────────────────────────────
with tab_search:
    query = st.text_input(
        "Enter your query",
        placeholder="e.g. What are the payment terms?",
        label_visibility="collapsed",
    )
    col_search, col_debug = st.columns([1, 5])
    search_clicked = col_search.button("Search", type="primary", use_container_width=True)
    debug_mode = col_debug.toggle("Debug view", value=False)

    if search_clicked and query.strip():
        with st.spinner("Searching..."):
            try:
                if debug_mode:
                    resp = requests.get(
                        f"{API_BASE}/search/debug",
                        params={"query": query, "k": options["top_k"]},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        render_debug(resp.json())
                    else:
                        st.error(f"Debug search failed: {resp.text}")
                else:
                    payload = {
                        "query": query,
                        "k": options["top_k"],
                        "mode": options["mode"],
                        "synthesize": options["synthesize"],
                    }
                    resp = requests.post(f"{API_BASE}/search", json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Log to history
                        st.session_state.query_history.append({
                            "query": query,
                            "latency_ms": data["latency_ms"],
                            "mode": data["mode"],
                            "results": len(data["results"]),
                        })

                        # Render answer if synthesis enabled
                        if options["synthesize"] and data.get("synthesis"):
                            render_answer(data["synthesis"])

                        # Render results
                        render_results(data["results"], data["latency_ms"], data["mode"])
                    else:
                        st.error(f"Search failed: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API — run `uvicorn neural_search.api.main:app --reload`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    elif search_clicked and not query.strip():
        st.warning("Please enter a query")

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    render_upload()

# ── History tab ───────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Recent Queries")
    history = st.session_state.query_history
    if not history:
        st.info("No queries yet — run a search first")
    else:
        for i, entry in enumerate(reversed(history[-20:]), start=1):
            st.markdown(
                f"`#{i}` &nbsp; **{entry['query']}** &nbsp;|&nbsp; "
                f"mode: `{entry['mode']}` &nbsp;|&nbsp; "
                f"results: `{entry['results']}` &nbsp;|&nbsp; "
                f"latency: `{entry['latency_ms']} ms`",
                unsafe_allow_html=True,
            )
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.rerun()
