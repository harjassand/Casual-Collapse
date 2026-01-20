from typing import Tuple

import torch
from torch import nn


class GraphInfer(nn.Module):
    def __init__(self, slot_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # slots: [B, K, D]
        b, k, d = slots.shape
        left = slots.unsqueeze(2).expand(b, k, k, d)
        right = slots.unsqueeze(1).expand(b, k, k, d)
        pair = torch.cat([left, right], dim=-1)
        logits = self.edge_mlp(pair).squeeze(-1)
        adj = torch.sigmoid(logits)
        eye = torch.eye(k, device=adj.device).unsqueeze(0)
        adj = adj * (1.0 - eye)
        return adj
