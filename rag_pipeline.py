# rag_pipeline.py

import os
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from gtts import gTTS
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

PDF_PATH = "job.pdf"


# --------------------------------------------------
# VECTOR STORE (BUILT ONCE)
# --------------------------------------------------
def build_vector_store_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num + 1},
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


vector_store = build_vector_store_from_pdf(PDF_PATH)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# --------------------------------------------------
# GEMINI LLM (STABLE MODEL)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


# --------------------------------------------------
# TEXT CLEANING (FOR TTS)
# --------------------------------------------------
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*#`>-]", "", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\n+", ". ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 400):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


# --------------------------------------------------
# ✅ MAIN FUNCTION (USED BY STREAMLIT)
# --------------------------------------------------
def answer_query(user_query: str):
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer the question using only the context below.

Rules:
1. Use numbers for main points.
2. Use lowercase letters for subpoints.
3. No markdown or special symbols.

Context:
{context}

Question:
{user_query}
"""

    response = llm.invoke(prompt)
    answer_text = response.content

    cleaned_text = clean_text_for_tts(answer_text)

    if not cleaned_text:
        cleaned_text = "Sorry, I could not generate a voice answer."

    audio_path = f"answer_{uuid.uuid4().hex}.mp3"

    chunks = chunk_text(cleaned_text)

    with open(audio_path, "wb") as f:
        for chunk in chunks:
            gTTS(text=chunk, lang="en").write_to_fp(f)

    return answer_text, audio_path
