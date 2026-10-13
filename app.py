import streamlit as st
import faiss
import re
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=None,
    layout="wide"
)


# ---------------- LOAD CUSTOM CSS ----------------
def load_css(file_name: str):
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")


# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">AI Research Assistant</h1>
        <p class="hero-subtitle">
            Upload a PDF and ask questions from its content.
        </p>
        <p class="hero-credit">Created by Muhammad Nouman</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return model


@st.cache_resource
def load_llm():
    model = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )
    return model


embedding_model = load_embedding_model()
generator = load_llm()


# ---------------- PDF PROCESSING ----------------
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    chunks = re.split(r'(?=\d+\.\s)', text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks


def create_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


# ---------------- RAG ----------------
def ask_rag(question, chunks, index):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    k = min(2, len(chunks))

    distances, indices = index.search(question_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False
    )

    answer = result[0]["generated_text"]

    return answer.strip(), retrieved_chunks


# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    if (
        "chunks" not in st.session_state
        or st.session_state.get("file_name") != uploaded_file.name
    ):
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(uploaded_file)
            index = create_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name

    st.success(
        f"PDF processed successfully! "
        f"{len(st.session_state.chunks)} chunks created."
    )

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Ask"):
        if question:
            with st.spinner("Searching and generating answer..."):
                answer, sources = ask_rag(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Context")

            for i, source in enumerate(sources):
                with st.expander(f"Source {i + 1}"):
                    st.write(source)
        else:
            st.warning("Please enter a question.")


# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        <p class="footer-credit">
            Powered by Hugging Face model
            <span class="model-name">Qwen2.5-0.5B-Instruct</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
