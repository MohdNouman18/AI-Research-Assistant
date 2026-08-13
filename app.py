
import streamlit as st
import faiss
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline



st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.write("Upload a PDF and ask questions from its content.")




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



def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

   
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    
    chunks = re.split(
        r'(?=\d+\.\s)',
        text
    )

   
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

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




def ask_rag(question, chunks, index):

   
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")


    k = min(2, len(chunks))

    
    distances, indices = index.search(
        question_embedding,
        k
    )

  
    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

   
    context = "\n\n".join(
        retrieved_chunks
    )

 
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



uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)



if uploaded_file:

    # Process only once
    if (
        "chunks" not in st.session_state
        or st.session_state.get("file_name") != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            chunks = process_pdf(
                uploaded_file
            )

            index = create_index(
                chunks
            )

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name

    st.success(
        f"PDF processed successfully! "
        f"{len(st.session_state.chunks)} chunks created."
    )

    question = st.text_input(
        "💬 Ask a question about your PDF:"
    )


    if st.button("Ask"):

        if question:

            with st.spinner(
                "Searching and generating answer..."
            ):

                answer, sources = ask_rag(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

         
            st.subheader("Answer")

            st.write(answer)


            # Retrieved context
            st.subheader("📚 Retrieved Context")

            for i, source in enumerate(sources):

                with st.expander(
                    f"Source {i + 1}"
                ):

                    st.write(source)

        else:

            st.warning(
                "Please enter a question."
            )
