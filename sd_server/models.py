"""Pipeline load/cache + generation for the SD server.

One pipeline — and its attached LoRA stack + scheduler — is kept in memory at
a time (single-GPU assumption). Switching any of them frees/reloads as
needed. A module-level lock serialises generation so concurrent requests
don't fight over the GPU — this matches ODT's row-by-row (sequential)
tagging loop.
"""
import inspect
import threading
import time

from capability import detect_capability

_lock = threading.Lock()
_loaded = {
    "model_id": None, "pipe": None, "device": "cpu",
    "lora_ids": [],            # ids of the currently-loaded LoRA stack, in order
    "scheduler_key": "default",
    "default_scheduler_cls": None,
    "default_scheduler_config": None,
}

# A curated set of well-known schedulers/samplers — the single biggest lever
# on output quality/speed after steps. "default" leaves whatever the pipeline
# shipped with (restored from the config cached at load time).
SCHEDULERS = {
    "default":         {"label": "Default (as shipped)"},
    "dpmpp_2m":        {"label": "DPM++ 2M", "cls": "DPMSolverMultistepScheduler"},
    "dpmpp_2m_karras": {"label": "DPM++ 2M Karras", "cls": "DPMSolverMultistepScheduler",
                       "kwargs": {"use_karras_sigmas": True}},
    "euler":           {"label": "Euler", "cls": "EulerDiscreteScheduler"},
    "euler_a":         {"label": "Euler Ancestral", "cls": "EulerAncestralDiscreteScheduler"},
    "ddim":            {"label": "DDIM", "cls": "DDIMScheduler"},
}


def get_loaded_model():
    return _loaded["model_id"]


def get_loaded_loras():
    return list(_loaded["lora_ids"])


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
    common = dict(torch_dtype=dtype, token=token or None)
    # The safety_checker weights aren't downloaded (see downloader.py), and
    # pipeline classes without one (SDXL/SD3/FLUX) don't accept the kwarg at
    # all — try with it disabled first, fall back for those that reject it.
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, use_safetensors=True, safety_checker=None,
            requires_safety_checker=False, **common,
        )
    except TypeError:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, use_safetensors=True, **common,
            )
        except Exception:
            # Some repos don't ship safetensors — retry without the flag.
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **common)

    if device == "cpu":
        pipe = pipe.to(device)
    else:
        # Stream weights layer-by-layer instead of materializing the whole
        # pipeline on-device at once — Flux/SD3.5-sized models can exceed
        # physical RAM, and on Apple Silicon "device" memory is the same
        # unified pool as system RAM, so a plain .to(device) has no headroom
        # to fall back on and gets killed by the OS under memory pressure.
        pipe.enable_sequential_cpu_offload(device=device)
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
        _loaded["lora_ids"] = []  # a fresh pipe has no LoRA attached
        _loaded["scheduler_key"] = "default"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    pipe, device = _load(model_id, token)
    # Cache the pipeline's original scheduler so "default" can be restored
    # after switching to something else.
    _loaded["default_scheduler_cls"] = type(pipe.scheduler)
    _loaded["default_scheduler_config"] = dict(pipe.scheduler.config)
    _loaded.update(model_id=model_id, pipe=pipe, device=device)
    return pipe


def _ensure_loras(pipe, loras, token=None):
    """Attach a stack of LoRAs (each {'id':..., 'scale':...}) on the loaded
    pipeline. Strengths are applied via set_adapters — changing scale alone
    doesn't require reloading weights, only a different *set* of LoRA ids does."""
    loras = [
        {"id": (l.get("id") or "").strip(), "scale": float(l.get("scale", 1.0))}
        for l in (loras or []) if (l.get("id") or "").strip()
    ]
    ids = [l["id"] for l in loras]

    if ids != _loaded.get("lora_ids"):
        if _loaded.get("lora_ids"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        for i, l in enumerate(loras):
            pipe.load_lora_weights(l["id"], adapter_name=f"lora{i}", token=token or None)
        _loaded["lora_ids"] = ids

    if loras:
        adapter_names = [f"lora{i}" for i in range(len(loras))]
        pipe.set_adapters(adapter_names, adapter_weights=[l["scale"] for l in loras])


def _ensure_scheduler(pipe, scheduler_key):
    scheduler_key = scheduler_key or "default"
    if _loaded.get("scheduler_key") == scheduler_key:
        return
    spec = SCHEDULERS.get(scheduler_key, SCHEDULERS["default"])
    if scheduler_key == "default" or "cls" not in spec:
        cls = _loaded["default_scheduler_cls"]
        config = _loaded["default_scheduler_config"]
    else:
        import diffusers
        cls = getattr(diffusers, spec["cls"])
        config = dict(pipe.scheduler.config)
        config.update(spec.get("kwargs", {}))
    pipe.scheduler = cls.from_config(config)
    _loaded["scheduler_key"] = scheduler_key


def generate(model_id, prompt, negative_prompt="", width=512, height=512,
             steps=30, guidance_scale=7.5, seed=-1, num_images=1, token=None,
             loras=None, scheduler=None):
    """Return (list_of_PIL_images, seed_used, elapsed_sec)."""
    import torch

    with _lock:
        pipe = ensure_loaded(model_id, token)
        _ensure_loras(pipe, loras, token)
        _ensure_scheduler(pipe, scheduler)
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
