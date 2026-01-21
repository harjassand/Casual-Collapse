import torch


def resolve_device(pref: str = "auto") -> torch.device:
    pref = (pref or "auto").lower()
    if pref in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref in ("mps", "metal"):
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref in ("cpu",):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
