"""
ingest.py — Ingestion pipeline for PDF, CSV, Image, Webpage.
"""

import os
import requests
import pandas as pd
import pytesseract
from PIL import Image
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.utils import get_embeddings, get_text_splitter, FAISS_INDEX_PATH

# Windows Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SUPPORTED = {".pdf", ".csv", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> list[Document]:
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["type"] = "pdf"
        doc.metadata["source"] = os.path.basename(file_path)
    print(f"  PDF: {len(docs)} pages loaded")
    return docs


def load_csv(file_path: str) -> list[Document]:
    """
    Universal CSV loader:
    1. One rich summary document with all statistics
    2. Row-level documents (batches of 10) for specific lookups
    """
    df = pd.read_csv(file_path)
    source = os.path.basename(file_path)
    headers = list(df.columns)
    docs = []

    # ── Summary document ──
    lines = [f"=== DATASET: {source} ==="]
    lines.append(f"Rows: {len(df)} | Columns: {', '.join(headers)}")
    lines.append("")

    # Numeric stats
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        lines.append("-- Numeric Stats --")
        for col in num_cols:
            s = df[col].dropna()
            if len(s):
                lines.append(f"{col}: mean={s.mean():.2f} | min={s.min():.2f} | max={s.max():.2f} | missing={df[col].isna().sum()}")

    # Categorical stats
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        lines.append("-- Categorical Stats --")
        for col in cat_cols:
            vc = df[col].value_counts().head(5).to_dict()
            lines.append(f"{col}: unique={df[col].nunique()} | top={vc} | missing={df[col].isna().sum()}")

    # Auto cross-tab for binary columns
    lines.append("-- Key Insights --")
    for col in df.columns:
        vals = df[col].dropna().unique()
        if set(vals).issubset({0, 1, True, False}):
            pos = df[col].isin([1, True]).sum()
            tot = df[col].notna().sum()
            lines.append(f"{col}: {pos}/{tot} positive ({pos/tot*100:.1f}%)")
            for cat in cat_cols:
                if df[cat].nunique() <= 10:
                    grp = df.groupby(cat)[col].agg(["sum","count"])
                    grp["pct"] = (grp["sum"]/grp["count"]*100).round(1)
                    insight = " | ".join(f"{idx}: {int(r['sum'])}/{int(r['count'])} ({r['pct']}%)" for idx, r in grp.iterrows())
                    lines.append(f"  {col} by {cat}: {insight}")

    docs.append(Document(
        page_content="\n".join(lines),
        metadata={"type": "csv", "source": source, "chunk_type": "summary"}
    ))

    # ── Individual row documents for precise lookups ──
    # Each row is its own document so FAISS can find exact matches
    for idx, row in df.iterrows():
        # Build a natural language row description
        row_parts = []
        for col, val in zip(headers, row.values):
            if pd.notna(val):
                row_parts.append(f"{col}: {val}")
        row_text = f"Record {idx+1} — " + " | ".join(row_parts)
        docs.append(Document(
            page_content=row_text,
            metadata={"type": "csv", "source": source, "chunk_type": "row", "row_index": idx+1}
        ))

    print(f"  CSV: 1 summary + {len(docs)-1} individual rows | {len(df)} rows total")
    return docs


def load_image(file_path: str) -> list[Document]:
    image = Image.open(file_path).convert("RGB")
    text = pytesseract.image_to_string(image).strip()
    if not text:
        text = "[No readable text found. Image may be a diagram or photo without text.]"
    print(f"  Image: {len(text)} chars extracted via OCR")
    return [Document(
        page_content=text,
        metadata={"type": "image", "source": os.path.basename(file_path)}
    )]


def load_webpage(url: str) -> list[Document]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    text = "\n".join(lines)
    if not text:
        raise ValueError(f"No readable content at: {url}")
    print(f"  Webpage: {len(text)} chars scraped")
    return [Document(
        page_content=text,
        metadata={"type": "webpage", "source": url}
    )]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def load_file(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".csv":
        return load_csv(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        return load_image(file_path)
    else:
        raise ValueError(f"Unsupported type: {ext}. Supported: PDF, CSV, PNG, JPG, JPEG")


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def load_index(embeddings):
    path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if os.path.exists(path):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def save_index(index):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    index.save_local(FAISS_INDEX_PATH)


# ---------------------------------------------------------------------------
# Ingestion entry points
# ---------------------------------------------------------------------------

def ingest_file(file_path: str) -> dict:
    raw_docs = load_file(file_path)
    return _store(raw_docs, os.path.basename(file_path))


def ingest_url(url: str) -> dict:
    raw_docs = load_webpage(url)
    return _store(raw_docs, url)


def _store(raw_docs: list, label: str) -> dict:
    embeddings = get_embeddings()
    splitter = get_text_splitter()
    chunks = splitter.split_documents(raw_docs)

    if not chunks:
        raise ValueError(f"No content extracted from: {label}")

    print(f"  Embedding {len(chunks)} chunks...")
    existing = load_index(embeddings)

    if existing is None:
        index = FAISS.from_documents(chunks, embeddings)
    else:
        new = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new)
        index = existing

    save_index(index)
    return {
        "status": "success",
        "file": label,
        "type": chunks[0].metadata.get("type", "unknown"),
        "chunks_created": len(chunks),
    }