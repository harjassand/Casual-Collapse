from typing import Optional, Tuple

import torch
from torch import nn


class SlotDynamics(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 128,
        use_graph: bool = False,
    ) -> None:
        super().__init__()
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.use_graph = use_graph

        input_dim = slot_dim + action_dim
        self.slot_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        if use_graph:
            self.msg_mlp = nn.Sequential(
                nn.Linear(slot_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, slot_dim),
            )
        else:
            self.msg_mlp = None

    def forward(
        self,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # z: [B, K, D]
        b, k, d = z.shape
        if self.use_graph and adj is not None:
            left = z.unsqueeze(2).expand(b, k, k, d)
            right = z.unsqueeze(1).expand(b, k, k, d)
            pair = torch.cat([left, right], dim=-1)
            msgs = self.msg_mlp(pair) * adj.unsqueeze(-1)
            msg = msgs.sum(dim=2)
        else:
            msg = torch.zeros_like(z)

        if self.action_dim == 0:
            action_in = torch.zeros(b, k, 0, device=z.device)
        elif action is None:
            action_in = torch.zeros(b, k, self.action_dim, device=z.device)
        else:
            action = action.view(b, -1)
            if action.shape[1] != self.action_dim:
                action = action[:, : self.action_dim]
            action_in = action.unsqueeze(1).expand(b, k, self.action_dim)

        inp = torch.cat([z + msg, action_in], dim=-1)
        delta = self.slot_mlp(inp)
        return z + delta

    def rollout(
        self,
        z0: torch.Tensor,
        actions: Optional[torch.Tensor],
        adj: Optional[torch.Tensor],
        horizon: int,
    ) -> torch.Tensor:
        z = z0
        outputs = []
        for t in range(horizon):
            action_t = None
            if actions is not None:
                action_t = actions[:, t]
            z = self.forward(z, action_t, adj)
            outputs.append(z)
        return torch.stack(outputs, dim=1)
