"""Detect this machine's compute capability for Stable Diffusion.

Reports the active backend (cuda / mps / cpu), device name, VRAM (or unified
RAM on Apple Silicon), total system RAM, and library versions. Everything is
wrapped in try/except so a missing dependency degrades gracefully to a
CPU-only report rather than crashing the server.
"""
import os
import platform


def _system_ram_mb():
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        pass
    try:  # POSIX fallback (Linux / macOS)
        return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 * 1024))
    except Exception:
        return 0


def detect_capability():
    info = {
        "backend":       "cpu",
        "device_name":   platform.processor() or platform.machine() or "CPU",
        "vram_total_mb": 0,
        "vram_free_mb":  0,
        "ram_mb":        _system_ram_mb(),
        "torch":         None,
        "diffusers":     None,
        "platform":      platform.platform(),
        "warning":       "",
    }

    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["backend"]       = "cuda"
            info["device_name"]   = torch.cuda.get_device_name(idx)
            info["vram_total_mb"] = int(props.total_memory / (1024 * 1024))
            try:
                free, _ = torch.cuda.mem_get_info(idx)
                info["vram_free_mb"] = int(free / (1024 * 1024))
            except Exception:
                info["vram_free_mb"] = info["vram_total_mb"]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # Apple Silicon: unified memory — report system RAM as the VRAM proxy.
            info["backend"]       = "mps"
            info["device_name"]   = "Apple Silicon (MPS)"
            info["vram_total_mb"] = info["ram_mb"]
            info["vram_free_mb"]  = info["ram_mb"]
        else:
            info["warning"] = "No GPU detected — generation will run on CPU and be very slow."
    except Exception as e:
        info["warning"] = f"torch unavailable: {e}"

    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except Exception:
        pass

    return info


def _cpu_ram_metrics():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "cpu_percent":  psutil.cpu_percent(interval=None),
            "ram_used_mb":  int(vm.used / (1024 * 1024)),
            "ram_total_mb": int(vm.total / (1024 * 1024)),
            "ram_percent":  vm.percent,
        }
    except Exception:
        return {"cpu_percent": None, "ram_used_mb": None, "ram_total_mb": None, "ram_percent": None}


def _nvidia_gpu_metrics():
    """Live utilization/temperature/VRAM via nvidia-ml-py — no sudo needed,
    but only installed/available on machines with an NVIDIA GPU (see
    requirements.txt). Returns None (fields report as unavailable) rather
    than raising if the library or a GPU isn't present."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            return {
                "utilization_percent": util.gpu,
                "vram_used_mb":        int(mem.used / (1024 * 1024)),
                "vram_total_mb":       int(mem.total / (1024 * 1024)),
                "temperature_c":       temp,
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _mps_gpu_metrics():
    """Apple Silicon has no public, sudo-free API for GPU utilization or
    temperature (powermetrics can report both but requires root, which this
    server won't request). VRAM-used is approximable via torch's own MPS
    allocator accounting — real, just PyTorch's view of it rather than the
    whole system's."""
    try:
        import torch
        used = torch.mps.driver_allocated_memory()
        return {
            "utilization_percent": None,
            "vram_used_mb":        int(used / (1024 * 1024)),
            "vram_total_mb":       None,  # filled in by caller from system RAM
            "temperature_c":       None,
        }
    except Exception:
        return None


def get_live_metrics():
    """Point-in-time system load for the tagging page's metrics panel.
    Cheap to call repeatedly (polled every few seconds) — no persistent
    state beyond what psutil/pynvml/torch already track internally."""
    base = _cpu_ram_metrics()
    cap  = detect_capability()

    gpu = {
        "backend":              cap["backend"],
        "device_name":          cap["device_name"],
        "utilization_percent":  None,
        "vram_used_mb":         None,
        "vram_total_mb":        cap.get("vram_total_mb") or None,
        "temperature_c":        None,
    }

    if cap["backend"] == "cuda":
        nv = _nvidia_gpu_metrics()
        if nv:
            gpu.update(nv)
        else:
            # torch already gave us total/free at capability-detection time.
            gpu["vram_used_mb"] = (cap.get("vram_total_mb") or 0) - (cap.get("vram_free_mb") or 0)
    elif cap["backend"] == "mps":
        mps = _mps_gpu_metrics()
        if mps:
            gpu.update(mps)
            gpu["vram_total_mb"] = cap.get("vram_total_mb")  # unified RAM, from capability

    return {**base, "gpu": gpu}
