"""
Cranfield Information Retrieval System
Interactive search interface.
"""

import streamlit as st
from src.retrieval.tfidf import TFIDFRetriever
from pathlib import Path
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.sentence_transformer import SentenceTransformerRetriever
from src.retrieval.hybrid import HybridRetriever


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RankFusion | Intelligent Search",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("🔎 RankFusion")
st.caption(
    "Intelligent Information Retrieval | "
    "TF-IDF • BM25 • Semantic Search • Hybrid RRF"
)
st.markdown(
    """
    Search the Cranfield aerodynamics collection using
    different information retrieval models.

    Compare traditional keyword-based retrieval with
    semantic search and hybrid retrieval.
    """
)

st.divider()


# --------------------------------------------------
# Search settings
# --------------------------------------------------

st.subheader("Search Configuration")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., aerodynamic performance of aircraft wings",
    )

with col2:
    top_k = st.selectbox(
        "Number of results",
        options=[5, 10, 15, 20],
        index=1,
    )

model_name = st.selectbox(
    "Select retrieval model",
    options=[
        "TF-IDF",
        "BM25",
        "Sentence Transformer",
        "Hybrid RRF",
    ],
)

search_button = st.button(
    "🔍 Search",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Search action placeholder
# --------------------------------------------------

# --------------------------------------------------
# Search action
# --------------------------------------------------

@st.cache_resource
def load_retriever(model_name: str):
    """Load and cache the selected retrieval model."""

    if model_name == "TF-IDF":
        model = TFIDFRetriever()

    elif model_name == "BM25":
        model = BM25Retriever()

    elif model_name == "Sentence Transformer":
        model = SentenceTransformerRetriever()
    elif model_name == "Hybrid RRF":
        model = HybridRetriever()
    else:
        raise ValueError(f"Unsupported retrieval model: {model_name}")

    model.fit()
    return model


# --------------------------------------------------
# Search action
# --------------------------------------------------

if search_button:
    if not query.strip():
        st.warning("Please enter a search query.")

    else:
        try:
            with st.spinner(f"Searching with {model_name}..."):
                retriever = load_retriever(model_name)
                results = retriever.search(query, top_k=top_k)

            st.subheader("Search Results")
            st.write(
                f"Found {len(results)} results for: **{query}**"
            )

            if not results:
                st.warning(
                    "No matching documents found. "
                    "Try a different query."
                )

            for result in results:
                with st.container(border=True):
                    st.markdown(
                        f"### {result['rank']}. "
                        f"{result.get('title', 'Untitled document')}"
                    )

                    st.caption(
                        f"Document ID: {result['document_id']} | "
                        f"{model_name} Score: {result['score']:.4f}"
                    )

                    st.write(result["text"])

        except Exception as exc:
            st.error(f"An error occurred during search: {exc}")

else:
    st.caption(
        "Enter a query, choose a retrieval model, and click Search."
    )