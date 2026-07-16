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
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
# Load from this file's own directory, not cwd, so HF_TOKEN is picked up
# whether the server is launched via `python app.py`, `cd sd_server && ...`,
# or scripts/setup_mac.sh.
load_dotenv(os.path.join(_SERVER_DIR, ".env"))

import downloader
import models as model_mgr
from capability import detect_capability, get_live_metrics

app = FastAPI(title="ODT Stable Diffusion Server")

CATALOG_PATH = os.path.join(_SERVER_DIR, "catalog.json")
LORA_CATALOG_PATH = os.path.join(_SERVER_DIR, "lora_catalog.json")


def load_catalog():
    try:
        with open(CATALOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def load_lora_catalog():
    try:
        with open(LORA_CATALOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/health")
def health():
    return {"status": "ok", "loaded_model": model_mgr.get_loaded_model()}


@app.get("/status")
def status():
    """Live activity — what the server is doing right now (loading weights,
    loading a LoRA, generating step N/M, idle) plus a short recent-events
    log, so a caller polling this can show real progress instead of a
    black box while /generate is in flight."""
    return model_mgr.get_status()


@app.post("/cancel")
def cancel():
    """Best-effort: interrupts an in-flight generation. See
    models.request_cancel for what this can and can't stop."""
    model_mgr.request_cancel()
    return {"cancelled": True}


@app.get("/capability")
def capability():
    return detect_capability()


@app.get("/metrics")
def metrics():
    """Live CPU/RAM/GPU load for the tagging page's system-metrics panel —
    separate from the static, one-time /capability report."""
    return get_live_metrics()


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


@app.get("/loras")
def list_loras():
    """LoRAs are plain HF repos downloaded via the same /download endpoint
    (kind='lora'); which ones are LoRAs is tracked in downloader's registry
    since that can't be inferred from the shared HF cache alone."""
    downloaded = downloader.downloaded_lora_ids()
    catalog = load_lora_catalog()
    catalog_ids = {l["id"] for l in catalog}

    out = []
    for l in catalog:
        entry = dict(l)
        entry["downloaded"] = l["id"] in downloaded
        out.append(entry)
    for rid in sorted(downloaded - catalog_ids):
        out.append({"id": rid, "label": rid, "gated": False, "downloaded": True,
                    "notes": "Downloaded via custom repo ID."})
    return {"loras": out}


class DownloadReq(BaseModel):
    model_id: str
    hf_token: Optional[str] = None
    kind: Optional[str] = "model"  # "model" or "lora"


@app.post("/download")
def download(req: DownloadReq):
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    kind = req.kind if req.kind in ("model", "lora") else "model"
    job_id = downloader.start_download(req.model_id, req.hf_token or os.environ.get("HF_TOKEN"), kind=kind)
    return {"job_id": job_id}


@app.get("/download/status")
def download_status(job_id: str):
    return downloader.status(job_id)


@app.get("/schedulers")
def list_schedulers():
    return {"schedulers": [{"key": k, "label": v["label"]} for k, v in model_mgr.SCHEDULERS.items()]}


class LoraSpec(BaseModel):
    id: str
    scale: float = 1.0


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
    loras: List[LoraSpec] = []
    scheduler: Optional[str] = "default"


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
            token=req.hf_token or os.environ.get("HF_TOKEN"),
            loras=[l.dict() for l in req.loras],
            scheduler=req.scheduler,
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
