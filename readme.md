#  Multi-Modal RAG System

A production-ready **Retrieval Augmented Generation (RAG)** system that ingests and queries information from multiple data types — PDF, CSV, Images (OCR), and Webpages — using a unified vector store and LLM-powered responses.


## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                  FastAPI + Dark Chat UI                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │         INGESTION           │
            │  PDF → PyMuPDF              │
            │  CSV → pandas + stats       │
            │  Image → Tesseract OCR      │
            │  Webpage → BeautifulSoup    │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │    EMBEDDING + STORAGE      │
            │  HuggingFace all-MiniLM-L6  │
            │  FAISS Vector Database      │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │     LLM-BASED ROUTING       │
            │  Groq LLaMA classifies      │
            │  query → best modality      │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │   RETRIEVAL + GENERATION    │
            │  FAISS similarity search    │
            │  Metadata filtering         │
            │  Groq LLaMA streaming       │
            └─────────────────────────────┘


## ✨ Features

- **4 Data Types** — PDF, CSV, Image (OCR), Webpage
- **Unified FAISS Index** — all data types in one vector store
- **LLM-Based Query Routing** — Groq LLaMA semantically decides which data type to search
- **Metadata Filtering** — filters retrieved chunks by document type
- **Streaming Responses** — real-time token-by-token output
- **Persistent Index** — FAISS index saved to disk, no re-indexing needed after restart
- **Dark Chat UI** — Claude-inspired interface with sidebar, chat history, file management

---

## 🚀 Advanced Features Implemented

| Feature | Description |
|---------|-------------|
| ✅ Query Routing | LLM classifies intent → routes to correct data type |
| ✅ Metadata Filtering | Retrieves chunks filtered by `doc.metadata["type"]` |
| ✅ Streaming Responses | Token-by-token streaming via Groq + FastAPI |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Python |
| Frontend | HTML/CSS/JS (dark chat UI) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector DB | FAISS |
| LLM | Groq LLaMA 3.1 8B Instant |
| PDF Loader | PyMuPDF |
| CSV Loader | pandas |
| Image OCR | Tesseract OCR |
| Web Scraper | BeautifulSoup4 |
| Framework | LangChain |


## 📁 Project Structure

Vaics/
├── main.py                  # FastAPI server — upload, index, stream
├── requirements.txt         # All dependencies
├── .env                     # API keys (not committed)
├── .gitignore
├── README.md
├── templates/
│   └── index.html           # Chat UI
└── src/
    ├── utils.py             # Embeddings, text splitter, config
    ├── ingest.py            # File loaders for all 4 data types
    ├── router.py            # LLM-based semantic query router
    └── query.py             # FAISS retrieval + Groq LLM generation


## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Shiva9966/multimodal-rag-system.git
cd multimodal-rag-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for image support)
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`

### 4. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at: https://console.groq.com

### 5. Run the server
```bash
python -m uvicorn main:app --reload --port 8000
```

### 6. Open browser
```
http://localhost:8000
```

---

## 💡 How to Use

1. **Upload a file** — Click "Upload file" in the sidebar (PDF, CSV, PNG, JPG)
2. **Index a webpage** — Paste any URL in the sidebar and click "Go"
3. **Ask questions** — Type in the chat input and press Enter
4. **Switch context** — Click any file in sidebar to focus queries on that file
5. **New chat** — Click "+ New" to start fresh (index persists)

---

## 📊 Example Queries

| Data Type | Example Query |
|-----------|--------------|
| PDF | *"What are the projects in the resume?"* |
| CSV | *"What is the survival rate by passenger class?"* |
| CSV | *"Who survived among female passengers?"* |
| Image | *"What components are shown in the diagram?"* |
| Webpage | *"What are the types of quantization techniques?"* |
| Cross-modal | *"Compare skills in resume with concepts on the webpage"* |

---

## 🔧 Design Decisions

### Why FAISS over ChromaDB?
- Zero external dependencies, runs fully local
- Fast similarity search even with large datasets
- Persistent index saved to disk automatically

### Why HuggingFace Embeddings over OpenAI?
- Completely free, no API calls for embedding
- Runs locally, no internet needed after first download
- `all-MiniLM-L6-v2` is fast and accurate for semantic search

### Why LLM-based routing over keyword matching?
- Handles synonyms: *"who perished"* → CSV, not just *"who died"*
- Language-agnostic, works for any domain
- No manual keyword maintenance needed

### Why Groq over OpenAI GPT?
- Free tier with high rate limits
- Ultra-fast inference (low latency streaming)
- LLaMA 3 quality matches GPT-3.5 for RAG tasks

---

## ⚠️ Known Tradeoffs

- **OCR limitation** — Images with complex diagrams return text labels only, not visual structure
- **LLM router adds latency** — ~0.5s extra per query for routing classification
- **Token limits** — Free Groq tier has daily token limits; upgrade for production use
- **PDF images ignored** — Embedded images inside PDFs are not extracted (text only)
