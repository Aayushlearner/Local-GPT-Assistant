# app.py

import io
from typing import List, Dict, Any

import numpy as np
import streamlit as st

# Load environment variables at startup
from dotenv import load_dotenv
load_dotenv()

from rag_engine.extractor import extract_text
from rag_engine.chunker import chunk_text
from rag_engine.embedder import EmbeddingModel
from rag_engine.vector_store import VectorStore
from rag_engine.generator import generate_answer_with_groq, FALLBACK_MESSAGE


st.set_page_config(page_title="Local RAG Assistant", page_icon="🧠", layout="wide")


def init_state():
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()
    if "vector_store" not in st.session_state:
        dim = st.session_state.embedding_model.embedding_dim
        st.session_state.vector_store = VectorStore(embedding_dim=dim)
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files: List[str] = []


def process_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    embedding_model: EmbeddingModel = st.session_state.embedding_model
    vector_store: VectorStore = st.session_state.vector_store

    for file in uploaded_files:
        st.write(f"📄 Processing: **{file.name}**")

        file_bytes = file.read()
        file_obj = io.BytesIO(file_bytes)

        try:
            text = extract_text(file_obj, file.name)
        except Exception as e:
            st.error(f"Failed to extract text from {file.name}: {e}")
            continue

        if not text.strip():
            st.warning(f"No text found in {file.name}, skipping.")
            continue

        chunks = chunk_text(text, chunk_size=350, overlap=50)
        if not chunks:
            st.warning(f"No chunks created for {file.name}, skipping.")
            continue

        embeddings = embedding_model.embed_texts(chunks)

        metadatas: List[Dict[str, Any]] = []
        for idx, _ in enumerate(chunks):
            metadatas.append({"file_name": file.name, "chunk_id": idx})

        vector_store.add(embeddings, chunks, metadatas)
        st.session_state.indexed_files.append(file.name)

    st.success("✅ All files processed and indexed.")


def main():
    init_state()

    st.title("🧠 Local RAG Assistant (Groq + FAISS)")
    st.markdown(
        """
This app lets you upload documents and ask questions **only based on those documents**.  
If the answer is not found, it will say:

> *"I don't know based on the provided documents."*
"""
    )

    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload files",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.sidebar.button("📥 Process & Index Files"):
            with st.spinner("Indexing documents..."):
                process_uploaded_files(uploaded_files)

    if st.session_state.indexed_files:
        st.sidebar.markdown("### Indexed Files:")
        for fname in set(st.session_state.indexed_files):
            st.sidebar.write(f"- {fname}")

    st.markdown("---")
    st.subheader("💬 Ask a Question")

    question = st.text_input("Enter your question:")
    top_k = st.slider("Chunks to retrieve:", min_value=1, max_value=10, value=5)
    distance_threshold = st.slider(
        "Max allowed distance:",
        0.1, 5.0, 1.5, 0.1,
        help="If distance is higher than this, answer: I don't know."
    )

    if st.button("🔍 Get Answer"):
        vector_store: VectorStore = st.session_state.vector_store
        embedding_model: EmbeddingModel = st.session_state.embedding_model

        if not question.strip():
            st.warning("Please enter a question.")
            return

        if vector_store.is_empty:
            st.warning("Upload and index documents first.")
            return

        with st.spinner("Searching context..."):
            query_embedding = embedding_model.embed_query(question)
            results = vector_store.search(query_embedding, top_k=top_k)

        if not results:
            st.info(FALLBACK_MESSAGE)
            return

        min_distance = min(r["distance"] for r in results)
        if min_distance > distance_threshold:
            st.info(FALLBACK_MESSAGE)
            return

        with st.spinner("Generating answer..."):
            answer = generate_answer_with_groq(question, results)

        st.markdown("### 🧾 Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        for r in results:
            meta = r["metadata"]
            st.markdown(
                f"- **{meta.get('file_name')}**, Chunk: `{meta.get('chunk_id')}`, Distance: `{r['distance']:.3f}`"
            )


if __name__ == "__main__":
    main()
