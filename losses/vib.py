from typing import Optional, Tuple

import torch


def categorical_kl(logits: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    # logits: [B, K, M]
    q = torch.softmax(logits, dim=-1)
    log_q = torch.log_softmax(logits, dim=-1)
    if prior is None:
        log_p = -torch.log(torch.tensor(logits.shape[-1], device=logits.device, dtype=logits.dtype))
    else:
        log_p = torch.log(prior + 1e-8)
    kl = (q * (log_q - log_p)).sum(dim=-1).mean()
    return kl


def vib_loss(kl_term: torch.Tensor, log_likelihood: torch.Tensor, beta: float) -> torch.Tensor:
    return kl_term - beta * log_likelihood


def code_entropy(usage: torch.Tensor) -> torch.Tensor:
    probs = usage / usage.sum().clamp_min(1.0)
    entropy = -(probs * (probs + 1e-8).log()).sum()
    return entropy
