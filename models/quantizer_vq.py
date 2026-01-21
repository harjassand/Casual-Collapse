from typing import Dict, Tuple

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int,
        dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_ema: bool = True,
        soft_temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.use_ema = use_ema
        self.soft_temp = soft_temp

        codebook = torch.randn(num_codes, dim)
        if use_ema:
            self.register_buffer("codebook", codebook)
            self.register_buffer("ema_counts", torch.zeros(num_codes))
            self.register_buffer("ema_means", codebook.clone())
        else:
            self.codebook = nn.Parameter(codebook)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # z: [B, K, D]
        b, k, d = z.shape
        flat = z.reshape(-1, d)
        codebook = self.codebook
        if self.use_ema and self.training:
            codebook = self.codebook.detach().clone()
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(distances, dim=1)
        logits = -distances / max(self.soft_temp, 1e-6)
        probs = torch.softmax(logits, dim=1)
        quantized = codebook[indices].view(b, k, d)
        if self.use_ema:
            quantized = quantized.detach()

        codebook_loss = ((quantized.detach() - z) ** 2).mean()
        commitment_loss = ((quantized - z.detach()) ** 2).mean()
        loss = codebook_loss + self.commitment_weight * commitment_loss

        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = torch.nn.functional.one_hot(indices, self.num_codes).float()
                counts = one_hot.sum(dim=0)
                self.ema_counts = self.ema_counts * self.decay + counts * (1 - self.decay)
                dw = one_hot.t() @ flat
                self.ema_means = self.ema_means * self.decay + dw * (1 - self.decay)
                n = self.ema_counts.sum()
                weights = (self.ema_counts + self.eps) / (n + self.num_codes * self.eps)
                self.codebook.copy_(self.ema_means / weights.unsqueeze(1))

        quantized_st = z + (quantized - z).detach()

        counts = torch.bincount(indices, minlength=self.num_codes).float()
        usage = counts / counts.sum().clamp_min(1.0)
        soft_usage = probs.mean(dim=0)
        perplexity = torch.exp(-(usage * (usage + 1e-8).log()).sum())

        stats = {
            "indices": indices.view(b, k),
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "usage": usage,
            "soft_usage": soft_usage,
        }
        return quantized_st, loss, stats
