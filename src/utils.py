"""
utils.py — Centralized configuration for embeddings, text splitting, and environment.

Design Decision:
- HuggingFace local embeddings (all-MiniLM-L6-v2) — zero rate limits, runs offline
- Groq LLaMA 3.3 70B for generation — fast, 14,400 free req/day, 128k context
- No embedding API calls = instant indexing regardless of file size
"""

import os
import time
import random
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Constants ---
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Embedding model — runs 100% locally, no API key, no rate limits
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_groq_api_key() -> str:
    """Retrieve and validate the Groq API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in .env file.")
    return api_key


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize local HuggingFace embedding model.

    all-MiniLM-L6-v2 is the industry-standard lightweight RAG embedding model:
    - 384 dimensions, fast inference, great semantic quality
    - Downloads once (~90MB), cached locally after first run
    - Zero API calls, zero rate limits, works fully offline
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Return a configured text splitter.
    1500 chunk size gives rich context per chunk without being too large.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )