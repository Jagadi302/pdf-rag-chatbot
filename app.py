
from pathlib import Path
import hashlib
import json
import urllib.request

import faiss
import numpy as np
import pymupdf
import streamlit as st


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DEFAULT_PDF = (
    BASE_DIR
    / "documents"
    / "AI_Basics_Cheat_Sheet_Jagadish.pdf"
)

st.title("📚 PDF RAG Chatbot")
st.caption(
    "Ask questions about a document and inspect the source pages."
)


# Generate embeddings using Ollama
def get_embedding(text):
    data = {
        "model": "nomic-embed-text",
        "prompt": text,
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embedding"]


# Split text into overlapping chunks
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Extract text page by page, preserve page numbers,
# generate embeddings, and build the FAISS index
@st.cache_resource(show_spinner=False)
def build_pdf_index(pdf_bytes):
    page_chunks = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:
        page_count = pdf.page_count

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            for chunk in split_text(text):
                page_chunks.append({
                    "page": page_number,
                    "text": chunk,
                })

    if not page_chunks:
        raise ValueError(
            "No selectable text was found. "
            "This PDF may need OCR."
        )

    vectors = []

    progress = st.progress(0, text="Generating embeddings...")

    for i, item in enumerate(page_chunks, start=1):
        vector = get_embedding(
            "search_document: " + item["text"]
        )
        vectors.append(vector)

        progress.progress(
            i / len(page_chunks),
            text=f"Embedding chunk {i}/{len(page_chunks)}",
        )

    progress.empty()

    vectors = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, page_chunks, page_count


# Generate an answer using the retrieved passages
def generate_answer(question, sources):
    context = "\n\n".join(
        f"[PDF page {item['page']}]\n{item['text']}"
        for item in sources
    )

    prompt = f"""
You answer questions using the supplied PDF passages.

Use only the supplied context. Do not invent facts.
If the answer is not supported by the passages, say:
"I could not find the answer in the provided PDF."

Give a clear, direct answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


# Retrieve relevant chunks from FAISS
def answer_question(question, index, chunks):
    question_vector = np.asarray(
        [
            get_embedding(
                "search_query: " + question
            )
        ],
        dtype="float32",
    )

    if question_vector.shape[1] != index.d:
        raise ValueError(
            "Embedding dimensions do not match the index."
        )

    faiss.normalize_L2(question_vector)

    k = min(3, index.ntotal)
    scores, indices = index.search(question_vector, k)

    sources = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            source = chunks[idx].copy()
            source["score"] = float(score)
            sources.append(source)

    if not sources:
        return (
            "No relevant passages were retrieved.",
            [],
        )

    answer = generate_answer(question, sources)

    return answer, sources


# Document upload
with st.sidebar:
    st.header("📄 Select your PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
    )

    st.caption(
        "The PDF is processed locally using your Ollama models."
    )


# Use the uploaded PDF or the existing sample PDF
try:
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        document_name = uploaded_file.name
    else:
        if not DEFAULT_PDF.exists():
            st.error(
                "Default PDF not found. Please upload a PDF."
            )
            st.stop()

        pdf_bytes = DEFAULT_PDF.read_bytes()
        document_name = DEFAULT_PDF.name

    document_key = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("document_key") != document_key:
        st.session_state["document_key"] = document_key
        st.session_state["messages"] = []

    with st.spinner("Preparing your PDF index..."):
        index, chunks, page_count = build_pdf_index(pdf_bytes)

except Exception as error:
    st.error(f"Could not process this PDF: {error}")
    st.stop()


# Display document information
with st.sidebar:
    st.success("Document ready")
    st.write(f"**File:** {document_name}")
    st.write(f"**Pages:** {page_count}")
    st.write(f"**Indexed chunks:** {index.ntotal}")
    st.write("**Embedding model:** nomic-embed-text")
    st.write("**Answer model:** llama3")


# Show earlier conversation messages and source references
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.expander("View source passages"):
                for item in message.get("sources", []):
                    st.markdown(
                        f"**PDF page {item['page']}** "
                        f"(similarity: {item['score']:.4f})"
                    )
                    st.write(item["text"])


# Chat input
question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state["messages"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating an answer..."):
                answer, sources = answer_question(
                    question,
                    index,
                    chunks,
                )

            st.markdown(answer)

            if sources:
                with st.expander("View source passages"):
                    for item in sources:
                        st.markdown(
                            f"**PDF page {item['page']}** "
                            f"(similarity: {item['score']:.4f})"
                        )
                        st.write(item["text"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Something went wrong: {error}")
