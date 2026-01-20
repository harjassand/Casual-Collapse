from typing import Tuple

import torch
from torch import nn


class SlotDecoder(nn.Module):
    def __init__(self, slot_dim: int, out_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, K, D]
        return self.decoder(z)
