import streamlit as st
import faiss
import re
import numpy as np
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=None,
    layout="wide"
)


# ---------------- LOAD CSS ----------------
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


# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return model


# ---------------- LOAD LLM (CPU, NO device_map) ----------------
@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm()


# ---------------- PDF PROCESSING ----------------
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Clean whitespace
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # Split into ~600 character chunks with 100 char overlap
    chunk_size = 600
    overlap = 100
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def create_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# ---------------- RAG ----------------
def ask_rag(question, chunks, index):
    # Encode query (normalized)
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Retrieve top-k chunks
    k = min(4, len(chunks))

    distances, indices = index.search(question_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    # ---------- Proper chat template ----------
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the answer is not in the context, say "
                "'The answer is not available in the document.' "
                "Be concise and direct."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )

    # Only take NEW tokens (skip prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip(), retrieved_chunks


# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


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

    question = st.text_input("Ask a question about your PDF:")

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
