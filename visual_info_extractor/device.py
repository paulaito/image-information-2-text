import subprocess

def get_device():
    try:
        # Try NVIDIA first
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except Exception:
        pass

    # Fallback: parse lspci to detect GPU names
    try:
        out = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL).decode().lower()
        if "nvidia" in out:
            return "cuda"
    except Exception:
        pass

    return "cpu"