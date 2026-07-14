"""Background model downloads + local-cache inspection.

Downloads run in daemon threads; their state lives in an in-memory dict keyed
by job id and is polled over HTTP. Gated repos (SD 3.5, FLUX.1-dev) require a
Hugging Face token, passed straight through to huggingface_hub.
"""
import os
import threading
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

# Diffusers model repos commonly carry multiple redundant weight formats in
# one repo: standalone .ckpt "pruned" checkpoints, .bin duplicates of the
# .safetensors we actually load, and a safety_checker we don't use. For
# stable-diffusion-v1-5 alone this is the difference between a ~5.5GB and a
# ~36GB download. Skip them all — the diffusers `from_pretrained` calls in
# models.py only ever need the *.safetensors component weights + configs.
_IGNORE_PATTERNS = [
    "*.ckpt", "*.bin", "*.pt", "*.msgpack", "*.onnx", "*.ot", "*.h5",
    "safety_checker/*",
]

_jobs = {}
_jobs_lock = threading.Lock()


def _set(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _run(job_id, model_id, token):
    from huggingface_hub import snapshot_download
    _set(job_id, state="downloading", message="Downloading model files…")
    try:
        snapshot_download(repo_id=model_id, token=token or None,
                          ignore_patterns=_IGNORE_PATTERNS)
        _set(job_id, state="complete", message="Download complete.")
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "401" in msg or "403" in msg or "gated" in low or "authoriz" in low or "restricted" in low:
            msg = ("Access denied. This is a gated model — accept its licence on "
                   "Hugging Face and provide a valid HF token. (" + msg + ")")
        _set(job_id, state="error", message=msg)


def start_download(model_id, token=None):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"state": "queued", "model_id": model_id,
                         "message": "Queued.", "progress": None}
    threading.Thread(target=_run, args=(job_id, model_id, token), daemon=True).start()
    return job_id


def status(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {"state": "unknown", "message": "No such job."}))


def downloaded_repo_ids():
    """Repo ids present in the local Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        return {repo.repo_id for repo in info.repos if repo.repo_type == "model"}
    except Exception:
        return set()
