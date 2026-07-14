"""Background model downloads + local-cache inspection.

Downloads run in daemon threads; their state lives in an in-memory dict keyed
by job id and is polled over HTTP. Gated repos (SD 3.5, FLUX.1-dev) require a
Hugging Face token, passed straight through to huggingface_hub.
"""
import threading
import uuid

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
        snapshot_download(repo_id=model_id, token=token or None)
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
