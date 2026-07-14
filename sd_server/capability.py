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
