"""Pipeline load/cache + generation for the SD server.

One pipeline is kept in memory at a time (single-GPU assumption). Switching
models frees the previous pipeline first. A module-level lock serialises
generation so concurrent requests don't fight over the GPU — this matches
ODT's row-by-row (sequential) tagging loop.
"""
import inspect
import threading
import time

from capability import detect_capability

_lock = threading.Lock()
_loaded = {"model_id": None, "pipe": None, "device": "cpu"}


def get_loaded_model():
    return _loaded["model_id"]


def _select_device_dtype():
    import torch
    backend = detect_capability()["backend"]
    if backend == "cuda":
        return "cuda", torch.float16
    if backend == "mps":
        return "mps", torch.float16
    return "cpu", torch.float32


def _load(model_id, token=None):
    import torch
    from diffusers import AutoPipelineForText2Image

    device, dtype = _select_device_dtype()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, token=token or None, use_safetensors=True,
        )
    except Exception:
        # Some repos don't ship safetensors — retry without the flag.
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, token=token or None,
        )

    pipe = pipe.to(device)
    for opt in ("enable_attention_slicing", "enable_vae_slicing"):
        try:
            getattr(pipe, opt)()
        except Exception:
            pass
    return pipe, device


def ensure_loaded(model_id, token=None):
    if _loaded["model_id"] == model_id and _loaded["pipe"] is not None:
        return _loaded["pipe"]

    # Free the previously-loaded pipeline before loading a new one.
    if _loaded["pipe"] is not None:
        _loaded["pipe"] = None
        _loaded["model_id"] = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    pipe, device = _load(model_id, token)
    _loaded.update(model_id=model_id, pipe=pipe, device=device)
    return pipe


def generate(model_id, prompt, negative_prompt="", width=512, height=512,
             steps=30, guidance_scale=7.5, seed=-1, num_images=1, token=None):
    """Return (list_of_PIL_images, seed_used, elapsed_sec)."""
    import torch

    with _lock:
        pipe = ensure_loaded(model_id, token)
        device = _loaded["device"]

        if seed is None or int(seed) < 0:
            seed = int(time.time() * 1000) % (2 ** 32)
        seed = int(seed)
        # MPS does not support generator device="mps" for manual_seed reliably.
        gen_device = "cpu" if device == "mps" else device
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        kwargs = {
            "prompt":                prompt,
            "negative_prompt":       negative_prompt or None,
            "width":                 int(width),
            "height":                int(height),
            "num_inference_steps":   int(steps),
            "guidance_scale":        float(guidance_scale),
            "num_images_per_prompt": int(num_images),
            "generator":             generator,
        }
        # Drop kwargs the specific pipeline doesn't accept (e.g. FLUX has no
        # negative_prompt; Turbo ignores guidance). Keeps one code path for all.
        allowed = set(inspect.signature(pipe.__call__).parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        start = time.time()
        result = pipe(**kwargs)
        elapsed = time.time() - start
        return result.images, seed, elapsed
