# RAG Demo (LangChain + FAISS)

Simple Retrieval-Augmented Generation demo using LangChain, HuggingFace embeddings, and FAISS.

## Overview

This repository contains a small demo that:
- Loads a PDF document
- Splits it into chunks
- Builds embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in a FAISS index
- Provides a Streamlit frontend (`frontend.py`) to upload a PDF or use the default `RAG.pdf`, ask a question, and get the top matching passage.

The core embedding and search logic lives in `main.py`.

## Requirements

- Python 3.9+ (tested on Windows)
- A virtual environment is recommended

Install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer cmd.exe:

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Run the app

Start the Streamlit frontend:

```powershell
streamlit run frontend.py
```

Then open the browser link Streamlit provides. The UI allows you to upload a PDF (or the app will use `RAG.pdf` if present), enter a question, and click the search button to retrieve the single top answer.

You can also run the retrieval core directly (example):

```powershell
python main.py
```

## Usage notes

- Upload a normal (text-based) PDF for best results. Scanned PDFs that only contain images will not produce text chunks; OCR is required first.
- If an uploaded document yields no text, the app will show a friendly error explaining that OCR is needed.
- The frontend returns only the single best match (top-1) from the FAISS similarity search.

## Troubleshooting

- If you see an embedding error pointing into FAISS, it usually means no text was extracted from the PDF. Try a different PDF or run OCR first.
- If Git shows your `venv/` folder as tracked, run:

```powershell
git rm -r --cached venv
git commit -m "Remove venv from repo"
```

The repository's `.gitignore` already includes common virtual environment names (`venv/`, `.venv/`, `env/`).

## Files

- `main.py` - core PDF loading, chunking, embedding, and search helpers
- `frontend.py` - Streamlit UI (upload PDF, ask question, show top answer)
- `RAG.pdf` - example/default PDF (optional)
- `requirements.txt` - Python dependencies

## Next steps / Ideas

- Add optional OCR (Tesseract) for scanned PDFs
- Add caching of built FAISS indexes per-document
- Add an option to return N results instead of top-1

---

If you'd like, I can also add a short `run.ps1` script to start the app or add instructions to pin dependency versions.
