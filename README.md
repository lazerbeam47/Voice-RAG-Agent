# Voice RAG Agent

A simple Streamlit app that answers questions about a document and speaks the answer.

## Features
- RAG over provided PDF/data
- Text answer and playable audio
- Secrets kept out of git via `.gitignore`

## Setup
1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file (not committed) with:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```

## Run
```bash
streamlit run app.py
```

## Notes
- `.env`, `.venv`, audio files (`*.mp3`), and indexes (`faiss_index/`) are ignored by git.
- Update `rag_pipeline.py` if you change model or data location.
