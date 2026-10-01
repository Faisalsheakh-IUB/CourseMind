"""
CourseMind — RAG-Based Educational Chatbot (MVP)
=================================================

A Streamlit app that lets a professor upload course materials (PDFs) and
lets students ask questions that are answered strictly from those
documents, using a fully free tech stack:

    Frontend   : Streamlit
    Framework  : LangChain
    LLM        : Google Gemini (langchain-google-genai)
    Embeddings : HuggingFace all-MiniLM-L6-v2 (runs locally, no API key)
    Vector DB  : FAISS (stored on local disk in ./faiss_index)

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Load variables from a local .env file (e.g. GOOGLE_API_KEY) into os.environ.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Folder where the FAISS vector index is persisted between app restarts.
FAISS_INDEX_DIR = "faiss_index"

# A free, local sentence-embedding model. Runs on CPU and needs no API key.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# The Gemini chat model used to generate answers.
# Google periodically retires older Gemini models, so the model name is
# read from an environment variable with a sensible default. If Google
# deprecates the default below, set GEMINI_MODEL_NAME in your .env file
# instead of editing this file. Current model list:
# https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")

CHUNK_SIZE = 1000        # characters per text chunk
CHUNK_OVERLAP = 200      # overlap between consecutive chunks
RETRIEVAL_K = 4          # number of chunks retrieved per student question

st.set_page_config(page_title="CourseMind — Course Q&A", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# 2. CACHED RESOURCES
#    st.cache_resource ensures these heavy objects are created ONCE per
#    server process rather than on every Streamlit rerun (Streamlit reruns
#    the whole script on every interaction).
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Load the free, local HuggingFace embedding model (downloaded once)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Create the Gemini chat model client.

    Raises:
        ValueError: if GOOGLE_API_KEY was never provided. Raising a plain
        ValueError (instead of letting the underlying library throw an
        opaque error) lets the calling code show a clear, friendly message.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to a .env file in the "
            "project root (see the setup instructions) before asking "
            "questions."
        )
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,  # low temperature -> factual, less "creative" answers
    )


# ---------------------------------------------------------------------------
# 3. DOCUMENT PROCESSING HELPERS (Professor's side)
# ---------------------------------------------------------------------------

def extract_chunks_from_pdfs(uploaded_pdfs):
    """
    Read every uploaded PDF, extract its raw text, and split it into
    overlapping chunks small enough to embed and retrieve accurately.

    Returns:
        chunks (list[str]): text of every chunk, across all PDFs.
        metadatas (list[dict]): parallel list recording which source file
            each chunk came from, so answers can cite a specific document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks, metadatas = [], []

    for pdf_file in uploaded_pdfs:
        reader = PdfReader(pdf_file)
        raw_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # extract_text() returns None for image-only pages
                raw_text += page_text + "\n"

        if not raw_text.strip():
            # Nothing extractable — e.g. a scanned PDF with no text layer.
            st.sidebar.warning(
                f"⚠️ Could not read any text from '{pdf_file.name}'. "
                "It may be a scanned/image-only PDF. Skipping it."
            )
            continue

        file_chunks = splitter.split_text(raw_text)
        chunks.extend(file_chunks)
        metadatas.extend({"source": pdf_file.name} for _ in file_chunks)

    return chunks, metadatas


def build_faiss_index(chunks, metadatas, embeddings) -> FAISS:
    """Embed every chunk and persist the resulting FAISS index to disk."""
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def load_faiss_index(embeddings):
    """
    Load a previously saved FAISS index from disk, if one exists.
    Returns None if no course materials have been processed yet.
    """
    index_file = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    if not os.path.exists(index_file):
        return None
    # allow_dangerous_deserialization=True is required by FAISS's save/load
    # format (it unpickles the docstore). This is safe here because the
    # only index files on disk are the ones this same app created.
    return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# 4. ANSWER GENERATION (Student's side)
# ---------------------------------------------------------------------------

# The prompt instructs Gemini to answer ONLY from the retrieved context,
# which is what keeps the chatbot "strictly grounded" in course materials.
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful teaching assistant for a university course. "
            "Answer the student's question using ONLY the information given "
            "in the course context below — do not use any outside knowledge "
            "and do not make anything up. If the context does not contain "
            "the answer, reply exactly with: "
            "\"I couldn't find an answer to that in the uploaded course "
            "materials.\"\n\n"
            "Course context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def answer_question(question: str):
    """
    Run the full RAG pipeline for one student question:
      1. Load the FAISS index (raises ValueError if the professor hasn't
         processed any documents yet).
      2. Retrieve the most relevant chunks for the question.
      3. Ask Gemini to answer using only those chunks.

    Returns:
        answer (str)
        sources (list[str]): unique source filenames used for the answer.
    """
    embeddings = get_embeddings_model()
    vectorstore = load_faiss_index(embeddings)

    if vectorstore is None:
        raise ValueError(
            "No course materials have been processed yet. Please ask your "
            "professor to upload the syllabus/slides in the sidebar and "
            "click 'Process Documents' first."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "I couldn't find an answer to that in the uploaded course materials.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    sources = sorted({doc.metadata.get("source", "Unknown") for doc in relevant_docs})

    llm = get_llm()  # may raise ValueError if GOOGLE_API_KEY is missing
    messages = ANSWER_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return response.content, sources


# ---------------------------------------------------------------------------
# 5. SIDEBAR — PROFESSOR'S PANEL
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("👩‍🏫 Professor's Panel")
    st.caption("Upload course materials so students can ask questions about them.")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF course materials (syllabus, slides, notes...)",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_pdfs:
            st.warning("Please upload at least one PDF before processing.")
        else:
            try:
                with st.spinner("Reading PDFs, creating embeddings, and building the index..."):
                    chunks, metadatas = extract_chunks_from_pdfs(uploaded_pdfs)

                    if not chunks:
                        st.error(
                            "No readable text was found in the uploaded PDF(s). "
                            "Please upload text-based PDFs (not scanned images)."
                        )
                    else:
                        embeddings = get_embeddings_model()
                        build_faiss_index(chunks, metadatas, embeddings)
                        st.success(
                            f"✅ Knowledge base is ready! Processed "
                            f"{len(uploaded_pdfs)} file(s) into {len(chunks)} chunks."
                        )
            except Exception as e:
                st.error(f"❌ Failed to process documents: {e}")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# 6. MAIN WINDOW — STUDENT'S CHAT INTERFACE
# ---------------------------------------------------------------------------

st.title("📚 CourseMind — Ask Your Course Materials")
st.caption(
    "Ask a question about the syllabus, slides, or notes your professor has "
    "uploaded. Answers are based strictly on those documents."
)

if not GOOGLE_API_KEY:
    st.error(
        "⚠️ GOOGLE_API_KEY is not set. Add it to a `.env` file in the project "
        "root before asking questions (see the setup instructions)."
    )

# Chat history is kept in st.session_state so it survives Streamlit's
# rerun-on-every-interaction behavior instead of vanishing on each reload.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render every past message first, so the full conversation is visible.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# The chat input box is pinned to the bottom of the page by Streamlit.
user_question = st.chat_input("Ask a question about your course materials...")

if user_question:
    # 1. Show and store the student's question immediately.
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Generate and show the assistant's answer.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(user_question)
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.write(f"- {source}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ValueError as e:
                # Expected, user-facing errors: missing API key or no
                # knowledge base processed yet.
                error_message = f"⚠️ {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            except Exception as e:
                # Anything unexpected: network issues, malformed API
                # responses, rate limits, etc.
                error_message = f"❌ Something went wrong while generating an answer: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
