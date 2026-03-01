import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.utils import get_embeddings, get_groq_api_key, FAISS_INDEX_PATH
from src.router import route_query, describe_route, Modality

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 8
TOP_K_CSV = 20
TOP_K_SUMMARY = 15  # For summarize queries fetch more chunks
MIN_RESULTS = 1

SUMMARY_KEYWORDS = {"summarize", "summary", "overview", "brief", "outline", "what is the report", "what does the report", "tell me about the report", "explain the report"}

def load_index() -> FAISS:
    path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if not os.path.exists(path):
        raise FileNotFoundError("No documents indexed yet. Please upload a file first.")
    return FAISS.load_local(FAISS_INDEX_PATH, get_embeddings(), allow_dangerous_deserialization=True)

def retrieve(index: FAISS, query: str, modality: Modality):
    # Use higher k for summarization queries
    is_summary = any(kw in query.lower() for kw in SUMMARY_KEYWORDS)
    if modality == Modality.CSV:
        top_k = TOP_K_CSV
    elif is_summary:
        top_k = TOP_K_SUMMARY
    else:
        top_k = TOP_K
    if modality != Modality.GENERAL:
        candidates = index.similarity_search(query, k=top_k * 4)
        filtered = [d for d in candidates if d.metadata.get("type") == modality.value]
        if modality == Modality.CSV:
            summary = [d for d in filtered if d.metadata.get("chunk_type") == "summary"]
            rows    = [d for d in filtered if d.metadata.get("chunk_type") != "summary"]
            filtered = (summary + rows)[:top_k]
        else:
            filtered = filtered[:top_k]
        if len(filtered) >= MIN_RESULTS:
            return filtered, True
    docs = index.similarity_search(query, k=top_k)
    return docs, False

def build_prompt(query: str, docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        dtype  = doc.metadata.get("type", "unknown").upper()
        source = doc.metadata.get("source", "")
        ctype  = doc.metadata.get("chunk_type", "")
        label  = f"[{dtype}{' | '+ctype if ctype else ''} | {source}]"
        parts.append(f"{label}\n{doc.page_content[:600]}")
    context = "\n\n---\n\n".join(parts)
    return f"""You are a document assistant. Answer using ONLY the context below.

Rules:
- For SUMMARIZE requests: give a comprehensive summary of ALL topics found in context. Never say the full document is not provided — just summarize everything you can see.
- For WEBPAGE: answer questions about concepts, techniques, definitions from the article.
- For CSV: use numbers and statistics.
- For PDF/resume: extract exact details.
- For IMAGE: describe what the OCR text says.
- Be specific and extract exact information from context.
- Never refuse to answer if relevant content exists in context.

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""

def stream_answer(query: str):
    modality = route_query(query)
    route_desc = describe_route(modality)
    index = load_index()
    docs, filtered = retrieve(index, query, modality)

    yield {
        "type": "meta",
        "modality": modality.value,
        "route_description": route_desc,
        "was_filtered": filtered,
        "source_documents": docs,
    }

    if not docs:
        yield {"type": "text", "content": "No relevant documents found."}
        return

    client = Groq(api_key=get_groq_api_key())
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": build_prompt(query, docs)}],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield {"type": "text", "content": content}