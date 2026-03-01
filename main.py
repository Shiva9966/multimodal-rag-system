import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingest import ingest_file, ingest_url
from src.query import stream_answer

app = FastAPI(title="Personal Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
FILES_REGISTRY = "indexed_files.json"


# ── Persist file list to disk ──
def load_file_registry() -> list[dict]:
    # If FAISS index doesn't exist, registry is invalid — clear it
    faiss_path = os.path.join("faiss_index", "index.faiss")
    if not os.path.exists(faiss_path):
        if os.path.exists(FILES_REGISTRY):
            os.remove(FILES_REGISTRY)
        return []
    if os.path.exists(FILES_REGISTRY):
        with open(FILES_REGISTRY, "r") as f:
            return json.load(f)
    return []

def save_file_registry(files: list[dict]):
    with open(FILES_REGISTRY, "w") as f:
        json.dump(files, f, indent=2)


# Load on startup
indexed_files: list[dict] = load_file_registry()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"_tmp_{file.filename}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)
        result = ingest_file(tmp_path)
        result["file"] = file.filename

        # Avoid duplicates
        indexed_files[:] = [f for f in indexed_files if f["file"] != file.filename]
        indexed_files.append(result)
        save_file_registry(indexed_files)

        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


@app.post("/index-url")
async def index_url(url: str = Form(...)):
    try:
        result = ingest_url(url)

        # Avoid duplicates
        indexed_files[:] = [f for f in indexed_files if f["file"] != url]
        indexed_files.append(result)
        save_file_registry(indexed_files)

        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def generate():
        try:
            gen = stream_answer(request.query)
            meta = next(gen)
            yield json.dumps({
                "type": "meta",
                "route": meta.get("route_description", ""),
                "sources": [
                    {
                        "type": d.metadata.get("type", ""),
                        "source": d.metadata.get("source", ""),
                        "preview": d.page_content[:300]
                    }
                    for d in meta.get("source_documents", [])
                ]
            }) + "\n"

            for item in gen:
                if item.get("type") == "text" and item.get("content"):
                    yield json.dumps({
                        "type": "text",
                        "content": item["content"]
                    }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )


@app.get("/files")
async def get_files():
    return {"files": indexed_files}


@app.post("/reset")
async def reset():
    """Delete FAISS index + file registry — full reset."""
    import shutil
    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        if os.path.exists(FILES_REGISTRY):
            os.remove(FILES_REGISTRY)
        indexed_files.clear()
        return {"success": True, "message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-test")
async def ask_test(request: AskRequest):
    try:
        gen = stream_answer(request.query)
        meta = next(gen)
        full_text = ""
        for item in gen:
            if item.get("type") == "text":
                full_text += item["content"]
        return {
            "route": meta.get("route_description", ""),
            "answer": full_text,
            "sources": len(meta.get("source_documents", []))
        }
    except Exception as e:
        return {"error": str(e)}