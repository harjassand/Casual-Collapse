from typing import Optional

import torch


def cross_jacobian_penalty(next_z: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # next_z, z: [B, K, D]
    b, k, d = z.shape
    penalty = 0.0
    for i in range(k):
        out = next_z[:, i].sum()
        grads = torch.autograd.grad(out, z, create_graph=True, retain_graph=True)[0]
        for j in range(k):
            if j == i:
                continue
            penalty = penalty + grads[:, j].pow(2).mean()
    return penalty / max(k - 1, 1)


def total_correlation_penalty(z: torch.Tensor) -> torch.Tensor:
    # z: [B, K, D]
    b, k, d = z.shape
    flat = z.view(b, -1)
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.t() @ flat) / max(b - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()
