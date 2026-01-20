from typing import Optional, Tuple

import torch
from torch import nn


class SlotEncoder(nn.Module):
    def __init__(
        self,
        input_type: str,
        in_dim: int,
        slot_dim: int,
        num_slots: int,
        image_channels: int = 3,
        recon: bool = False,
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.recon = recon

        if input_type == "structured":
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, slot_dim),
                nn.ReLU(),
                nn.Linear(slot_dim, slot_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(slot_dim, slot_dim),
                nn.ReLU(),
                nn.Linear(slot_dim, in_dim),
            ) if recon else None
        elif input_type == "image":
            self.cnn = nn.Sequential(
                nn.Conv2d(image_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.proj = nn.Conv2d(64, slot_dim, 1)
            self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim))
            self.decoder = nn.Sequential(
                nn.Conv2d(slot_dim, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, image_channels, 1),
            ) if recon else None
        else:
            raise ValueError(f"Unknown input_type {input_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.input_type == "structured":
            # x: [B, K, D]
            slots = self.encoder(x)
            attn = torch.ones(x.shape[0], self.num_slots, 1, device=x.device) / self.num_slots
            recon = self.decoder(slots) if self.recon else None
            return slots, attn, recon

        # image mode
        features = self.cnn(x)
        proj = self.proj(features)  # [B, D, H, W]
        b, d, h, w = proj.shape
        proj_flat = proj.view(b, d, h * w).permute(0, 2, 1)  # [B, HW, D]
        queries = self.slot_queries.unsqueeze(0).expand(b, -1, -1)  # [B, K, D]
        attn_logits = torch.einsum("bkd,bnd->bkn", queries, proj_flat) / (self.slot_dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)  # [B, K, HW]
        slots = torch.einsum("bkn,bnd->bkd", attn, proj_flat)
        attn_maps = attn.view(b, self.num_slots, h, w)
        recon = None
        if self.recon:
            recon = self.decoder(slots.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))
        return slots, attn_maps, recon
