# src/api.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import faiss                
import uvicorn                   # import faiss before torch
faiss.omp_set_num_threads(1)

import io, json, threading, time
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image
import clipindex as ci

app = FastAPI(title="Dashcam retrieval API")
_lock = threading.Lock()                       # one embedding at a time on the GPU
_indexes, _meta = {}, []

@app.on_event("startup")
def load():
    global _meta
    _meta = [json.loads(l) for l in open(f"{ci.OUT}/meta.jsonl")]
    for name in ("flat", "sq8", "pq64"):
        p = f"{ci.OUT}/{name}.faiss"
        if os.path.exists(p):
            _indexes[name] = faiss.read_index(p)
    if not _indexes:
        raise RuntimeError("no index found - run build and finalize first")

class Query(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    k: int = Field(10, ge=1, le=50)
    index: str = "flat"

def _rank(vec, k, index):
    ix = _indexes.get(index)
    if ix is None:
        raise HTTPException(400, f"index '{index}' not available; have {list(_indexes)}")
    t = time.perf_counter()
    D, I = ix.search(vec, min(500, ix.ntotal))
    ms = 1000 * (time.perf_counter() - t)
    best = {}
    for s, i in zip(D[0], I[0]):
        if i >= 0 and _meta[i]["clip"] not in best:    # sorted, so first hit is the best frame
            best[_meta[i]["clip"]] = {"clip": _meta[i]["clip"], "t": _meta[i]["t"], "score": round(float(s), 4)}
    return {"results": list(best.values())[:k], "index": index, "search_ms": round(ms, 2)}

@app.get("/health")
def health():
    return {"status": "ok", "indexes": list(_indexes), "vectors": len(_meta),
            "clips": len({m["clip"] for m in _meta})}

@app.post("/predict")
def predict(q: Query):
    with _lock:
        vec = ci.embed_text([q.query])
    return {"query": q.query, **_rank(vec, q.k, q.index)}

@app.post("/predict/image")
def predict_image(file: UploadFile = File(...), k: int = 10, index: str = "flat"):
    try:
        img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read image")
    with _lock:
        vec = ci.embed_images([img])
    return _rank(vec, min(max(k, 1), 50), index)


if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)