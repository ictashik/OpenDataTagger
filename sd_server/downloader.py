"""Background model downloads + local-cache inspection.

Downloads run in daemon threads; their state lives in an in-memory dict keyed
by job id and is polled over HTTP. Gated repos (SD 3.5, FLUX.1-dev) require a
Hugging Face token, passed straight through to huggingface_hub.
"""
import json
import os
import threading
import time
import uuid

# huggingface_hub's per-chunk stall timeout defaults to 10s, which large
# weight files on a slow/congested link routinely exceed. Must be set before
# huggingface_hub is first imported anywhere in the process (it reads this
# into a module-level constant at import time) — this module is imported
# before any lazy `from huggingface_hub import ...` below, so set it here.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# The hf_xet accelerated-transfer backend (auto-used whenever the hf_xet
# package is installed) has its own network stack that does NOT honor
# HF_HUB_DOWNLOAD_TIMEOUT above — a stalled chunk can hang indefinitely with
# zero CPU use and no error. Disable it so downloads fall back to plain HTTP
# GETs, which do respect the timeout and fail (then retry/report) instead of
# hanging forever. Must be set before huggingface_hub is first imported, same
# reasoning as above.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Fully import huggingface_hub (and the requests/charset_normalizer/urllib3
# chain underneath it) right here in the main thread, before uvicorn starts
# dispatching requests to worker threads. FastAPI runs sync endpoints
# (download/generate) in a thread pool, so if this stays a lazy import inside
# those functions, two requests can race on the *first* import of the same
# module across threads — Python then returns a half-initialized module to
# whichever thread loses, raising spurious "module X has no attribute Y"
# errors (seen for both charset_normalizer and requests.exceptions).
import huggingface_hub  # noqa: F401

# diffusers' folder-based `from_pretrained` (used by models.py for base
# models) only ever reads: the small per-component configs (*.json), tokenizer
# text/vocab files, and each component's *.safetensors — always inside a named
# subfolder (unet/, vae/, text_encoder[_2]/, transformer/, tokenizer[_2]/,
# scheduler/, feature_extractor/). It never reads a *root-level* weight file —
# that's always the separate "single combined checkpoint" format for the
# original webui (e.g. stable-diffusion-v1-5 ships v1-5-pruned(.emaonly).
# (ckpt|safetensors) at its root, ~12GB on its own). Allow-listing to
# "safetensors only inside a subfolder" categorically excludes those
# regardless of what they're named, in any repo. Combined with skipping the
# unused safety_checker, this is the difference between a ~5GB and a ~36GB
# download of stable-diffusion-v1-5.
#
# LoRA repos are the opposite: their one weight file is conventionally at the
# repo ROOT (e.g. pytorch_lora_weights.safetensors), so this allow-list is
# only applied for kind="model" downloads, never for LoRAs.
_MODEL_ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "*/*.safetensors"]
_IGNORE_PATTERNS = ["safety_checker/*"]

_jobs = {}
_jobs_lock = threading.Lock()

# LoRAs are plain HF repos too (same cache, same snapshot_download call) but
# there's no reliable way to tell "this cached repo is a LoRA" apart from "a
# small base model" just by scanning the cache. Track LoRA repo ids we've
# downloaded explicitly in a small registry file next to this module.
_LORA_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_registry.json")
_registry_lock = threading.Lock()


def _load_lora_registry():
    try:
        with open(_LORA_REGISTRY_PATH) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _add_to_lora_registry(repo_id):
    with _registry_lock:
        ids = _load_lora_registry()
        ids.add(repo_id)
        with open(_LORA_REGISTRY_PATH, "w") as f:
            json.dump(sorted(ids), f)


def downloaded_lora_ids():
    return _load_lora_registry()


def _set(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _make_progress_tqdm(job_id):
    """A tqdm subclass fed to snapshot_download's `tqdm_class` hook. HF reuses
    this one class for several different progress bars concurrently:

    - the outer 'Fetching N files' bar (unit="it", one update per finished file)
    - a 'Downloading bytes' transfer bar (unit="B", total is an unreliable
      dedup/compression estimate — HF's own comment says not to trust it)
    - a 'Reconstructing' bar (unit="B", total = real sum of file sizes on the
      repo, updated as bytes are written to disk)

    Byte-level progress (%, GB/GB, speed, ETA) must come from the
    'Reconstructing' bar — it's the only one with a trustworthy total. The
    file-count bar still feeds the supplementary 'N/M files' readout."""
    from tqdm import tqdm as _tqdm

    class JobProgressTqdm(_tqdm):
        def update(self, n=1):
            result = super().update(n)
            total = self.total or 0
            done = self.n
            if self.unit == "B":
                if (self.desc or "").startswith("Reconstructing"):
                    _update_byte_progress(job_id, done, total)
            else:
                _set(job_id, files_done=done, files_total=total,
                    message=f"Fetching files… ({done}/{total})" if total else "Preparing download…")
            return result

    return JobProgressTqdm


def _update_byte_progress(job_id, done, total):
    """Update byte-level progress plus a smoothed download speed and ETA.

    Speed is an EMA of instantaneous rate, recomputed at most every 0.5s so
    frequent small chunk updates (10MB DOWNLOAD_CHUNK_SIZE) don't make the
    reading jump around.
    """
    now = time.monotonic()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        prev_ts = job.get("_speed_ts")
        prev_bytes = job.get("_speed_bytes", 0)
        speed = job.get("speed_bps")
        if prev_ts is None:
            job["_speed_ts"] = now
            job["_speed_bytes"] = done
        else:
            dt = now - prev_ts
            if dt >= 0.5:
                inst = max(0, done - prev_bytes) / dt
                speed = inst if speed is None else (0.3 * inst + 0.7 * speed)
                job["_speed_ts"] = now
                job["_speed_bytes"] = done
        pct = int(done / total * 100) if total else None
        eta = int((total - done) / speed) if (speed and total and done < total) else None
        job.update({
            "progress": pct,
            "bytes_done": done,
            "bytes_total": total,
            "speed_bps": speed,
            "eta_seconds": eta,
        })


def _run(job_id, model_id, token, kind):
    from huggingface_hub import snapshot_download
    _set(job_id, state="downloading", message="Downloading model files…", progress=0)
    kwargs = {
        "repo_id": model_id, "token": token or None,
        "ignore_patterns": _IGNORE_PATTERNS,
        "tqdm_class": _make_progress_tqdm(job_id),
    }
    if kind != "lora":
        kwargs["allow_patterns"] = _MODEL_ALLOW_PATTERNS
    try:
        snapshot_download(**kwargs)
        if kind == "lora":
            _add_to_lora_registry(model_id)
        _set(job_id, state="complete", message="Download complete.", progress=100)
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "401" in msg or "403" in msg or "gated" in low or "authoriz" in low or "restricted" in low:
            msg = ("Access denied. This is a gated model — accept its licence on "
                   "Hugging Face and provide a valid HF token. (" + msg + ")")
        _set(job_id, state="error", message=msg)


def start_download(model_id, token=None, kind="model"):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"state": "queued", "model_id": model_id, "kind": kind,
                         "message": "Queued.", "progress": None,
                         "files_done": 0, "files_total": 0,
                         "bytes_done": 0, "bytes_total": 0,
                         "speed_bps": None, "eta_seconds": None}
    threading.Thread(target=_run, args=(job_id, model_id, token, kind), daemon=True).start()
    return job_id


def status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id, {"state": "unknown", "message": "No such job."})
        return {k: v for k, v in job.items() if not k.startswith("_")}


def downloaded_repo_ids():
    """Repo ids present in the local Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return {repo.repo_id for repo in info.repos if repo.repo_type == "model"}
    except Exception:
        return set()
