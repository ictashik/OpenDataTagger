"""ODT Stable Diffusion server.

A thin HTTP wrapper around HuggingFace `diffusers` so ODT can generate images
the same way it talks to Ollama for text — over a plain HTTP port. Run it on
localhost for "local" generation or on a LAN GPU box for "remote"; ODT only
needs the host:port.

    pip install -r requirements.txt
    python app.py --port 7860          # or: uvicorn app:app --host 0.0.0.0 --port 7860
"""
import argparse
import base64
import io
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import downloader
import models as model_mgr
from capability import detect_capability

app = FastAPI(title="ODT Stable Diffusion Server")

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")


def load_catalog():
    try:
        with open(CATALOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/health")
def health():
    return {"status": "ok", "loaded_model": model_mgr.get_loaded_model()}


@app.get("/capability")
def capability():
    return detect_capability()


@app.get("/models")
def list_models():
    cap = detect_capability()
    vram = cap.get("vram_total_mb", 0) or 0
    downloaded = downloader.downloaded_repo_ids()
    catalog = load_catalog()
    catalog_ids = {m["id"] for m in catalog}

    out = []
    for m in catalog:
        entry = dict(m)
        entry["downloaded"] = m["id"] in downloaded
        # Unknown VRAM (e.g. CPU-only) -> don't block; flag runnable but slow.
        entry["runnable"] = (vram >= m.get("min_vram_mb", 0)) if vram else True
        out.append(entry)

    # Surface locally-cached models that aren't in the curated catalog.
    for rid in sorted(downloaded - catalog_ids):
        out.append({
            "id": rid, "label": rid, "pipeline_class": "?",
            "min_vram_mb": 0, "gated": False, "downloaded": True, "runnable": True,
            "default_width": 512, "default_height": 512,
            "default_steps": 30, "default_guidance": 7.5,
            "notes": "Found in local cache (not in curated catalog).",
        })

    return {"capability": cap, "models": out}


class DownloadReq(BaseModel):
    model_id: str
    hf_token: Optional[str] = None


@app.post("/download")
def download(req: DownloadReq):
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    job_id = downloader.start_download(req.model_id, req.hf_token)
    return {"job_id": job_id}


@app.get("/download/status")
def download_status(job_id: str):
    return downloader.status(job_id)


class GenerateReq(BaseModel):
    model_id: str
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    num_images: int = 1
    hf_token: Optional[str] = None


@app.post("/generate")
def generate(req: GenerateReq):
    if not req.model_id or not req.prompt:
        raise HTTPException(status_code=400, detail="model_id and prompt are required")
    try:
        images, seed_used, elapsed = model_mgr.generate(
            model_id=req.model_id,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            width=req.width,
            height=req.height,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            num_images=req.num_images,
            token=req.hf_token,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    encoded = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    return {"images": encoded, "seed_used": seed_used, "elapsed_sec": round(elapsed, 3)}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="ODT Stable Diffusion server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
