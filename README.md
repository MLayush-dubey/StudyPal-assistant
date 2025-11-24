# StudyPal Assistant

A modular Retrieval Augmented Generation (RAG) assistant for Class 12 STEM subjects, combining semantic search over textbooks/chapters and conversational AI. Designed for PDF-based knowledge ingestion, vector storage (ChromaDB), and interactive Q&A with Streamlit.

## Features

- **Conversational AI:** Uses Groq LLM and streaming chat memory for contextual, multi-turn Q&A.
- **Document Ingestion:** PDF chapters vectorized using HuggingFace embeddings.
- **Semantic Search:** ChromaDB allows deep, similarity-based text retrieval.
- **YouTube Integration:** Suggests relevant videos for study topics.
- **Extensible UI:** Streamlit front-end with dropdowns for subjects/chapters.

## Project Structure

```
StudyPal-assistant/
├── data/
│   └── class_12/
│       └── biology/
│           ├── 1.Life_Process.pdf
│           └── 2.Reproduction.pdf
├── vector_db/
├── chapters_vector_db/
├── src/
│   ├── main.py                # Streamlit web app
│   ├── chatbot_utility.py     # Chapter list logic
│   ├── get_yt_video.py        # YouTube search integration
│   ├── vectorize_book.py      # Vectorization pipeline
│   └── vectorize_script.py    # CLI for vectorization
├── .env
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/MLayush-dubey/StudyPal-assistant.git
cd StudyPal-assistant
python -m venv .venv
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the root:
```
GROQ_API_KEY=YOUR_API_KEY
CLASS_SUBJECT_NAME=biology
DEVICE=cuda      # or 'cpu' if you have a low-end/CPU-only setup
```

### 3. Add PDFs

Place your subject PDFs inside:
```
data/class_12/biology/
```

### 4. Vectorize

```bash
cd src
python vectorize_script.py
```

### 5. Launch the App

```bash
streamlit run main.py
```

## Note

I have only included **one subject (Biology)** with **two chapters** in my database, since I use a low-end GPU.  
If you have a more capable GPU, **add more subjects (e.g. Physics, Chemistry) and chapters** for broader experimentation and practice.  
In the Streamlit app, you’ll see Chemistry and Physics in the dropdown—these are currently **empty placeholders**. Feel free to add PDFs for these subjects if you want to expand or test the system.

## Main Components

- **main.py:** Streamlit chat UI, retrieval chain setup, search & memory features, YouTube recommendations.
- **chatbot_utility.py:** Chapter list logic per subject.
- **get_yt_video.py:** Fetches top-3 related YouTube videos using keywords.
- **vectorize_book.py / vectorize_script.py:** Prepares the vector DB for semantic search.

## Hardware Notes

- Supports both CPU and CUDA-based inference.
- For large-scale ingestion/vectorization, a more powerful GPU is recommended.

Feel free to copy, edit, or extend this according to your evolving project!
