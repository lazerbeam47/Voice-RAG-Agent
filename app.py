import streamlit as st
from rag_pipeline import answer_query
import os

st.set_page_config(
    page_title="Voice RAG",
    page_icon="🎤",
    layout="centered"
)

# session state
if "answer_text" not in st.session_state:
    st.session_state.answer_text = None

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

st.title("🎤 Voice-Enabled RAG App")
st.write("Ask a question about the document and hear the answer.")

user_query = st.text_input(
    "Enter your question",
    placeholder="e.g. What do we need to build in the assignment?"
)

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking and generating voice answer..."):
            answer_text, audio_path = answer_query(user_query)

        st.session_state.answer_text = answer_text
        st.session_state.audio_path = audio_path

# display answer
if st.session_state.answer_text:
    st.subheader("📄 Answer")
    st.write(st.session_state.answer_text)

# display audio (BYTES ✅)
if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
    st.subheader("🔊 Voice Answer")
    with open(st.session_state.audio_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")
