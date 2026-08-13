
# 🤖 AI Research Assistant

A simple RAG-based application that lets users upload PDF documents and ask questions about their content.

## Features

- PDF upload and text extraction
- Text chunking
- Semantic embeddings
- FAISS similarity search
- Qwen LLM-based answers
- Streamlit interface

## RAG Pipeline

PDF → Text Extraction → Chunking → Embeddings → FAISS → Context Retrieval → Qwen LLM → Answer

## Technologies

- Python
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyPDF
- PyTorch

## Models

**Embedding:** `sentence-transformers/all-MiniLM-L6-v2`

**LLM:** `Qwen/Qwen2.5-0.5B-Instruct`

## How It Works

1. Upload a PDF.
2. Extract and split the text.
3. Generate embeddings and store them in FAISS.
4. Ask a question.
5. Retrieve relevant content.
6. Generate an answer using Qwen.

## Purpose

A beginner friendly AI Research Assistant demonstrating a complete RAG pipeline for document based question answering.
