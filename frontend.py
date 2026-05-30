import tempfile
from pathlib import Path

import streamlit as st
from main import PDF_PATH, build_vector_store as build_core_vector_store

st.set_page_config(
    page_title="RAG Demo",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    h1, h2, h3, p, label {
        text-align: center;
    }

    div[data-testid="stFileUploader"] {
        max-width: 520px;
        margin: 0 auto;
    }

    div[data-testid="stTextInput"] {
        max-width: 620px;
        margin: 0 auto;
    }

    div[data-testid="stButton"] {
        display: flex;
        justify-content: center;
    }

    div[data-testid="stButton"] button {
        min-width: 180px;
    }

    .answer-box {
        padding: 1rem;
        border-radius: 12px;
        background-color: #f0f7ff;
        border-left: 5px solid #4a90e2;
        text-align: left;
        max-width: 760px;
        margin: 0 auto;
    }
    .top-answer {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(74,144,226,0.08);
        border: 1px solid rgba(74,144,226,0.12);
        max-width: 820px;
        margin: 0 auto 1rem auto;
        text-align: left;
        font-size: 16px;
        line-height: 1.5;
    }
    .top-answer h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #0b3d91;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# ---------- Vector Store ----------
@st.cache_resource
def get_vector_store(pdf_bytes: bytes, file_name: str):
    suffix = Path(file_name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    return build_core_vector_store(temp_path)


# ---------- Header ----------
st.title("📚 Retrieval Augmented Generation")

st.divider()

# ---------- Upload + Settings ----------
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

top_k = 1

# ---------- Load PDF ----------
if uploaded_file is not None:
    pdf_name = uploaded_file.name
    pdf_bytes = uploaded_file.getvalue()


else:
    pdf_name = PDF_PATH.name

    if not PDF_PATH.exists():
        st.error(f"Could not find {PDF_PATH.name}")
        st.stop()

    with open(PDF_PATH, "rb") as file:
        pdf_bytes = file.read()


st.divider()

# ---------- Question ----------
with col2:
    query = st.text_input(
        "Ask a question",
        placeholder="What are the three stages of a RAG pipeline?"
    )

with col2:
    search_clicked = st.button(
        "🔍 Search",
        type="primary"
    )

# ---------- Search ----------
if search_clicked:

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Building vector store and searching..."):
        try:
            db = get_vector_store(pdf_bytes, pdf_name)
            results = db.similarity_search(query=query, k=top_k)
        except ValueError as error:
            st.error(str(error))
            st.stop()

    # ---------- Quick Answer ----------
    st.subheader("💡 Top Answer")

    if results:
        st.markdown(
            f"""
            <div class="top-answer">
                <div>{results[0].page_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No answer found.")