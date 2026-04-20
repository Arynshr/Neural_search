import streamlit as st
import requests

API_BASE = "http://localhost:8000"


def render_upload():
    """Document upload widget with ingestion status."""
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF or DOCX files here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
            files = [
                ("files", (f.name, f.getvalue(), f.type))
                for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✓ Ingested {data['chunks_indexed']} chunks "
                        f"from {data['files_processed']} file(s)"
                    )
                    if data.get("warnings"):
                        for w in data["warnings"]:
                            st.warning(w)
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")
