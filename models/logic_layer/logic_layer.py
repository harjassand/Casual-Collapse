from typing import Dict, List, Optional

import torch
from torch import nn


class PredicateNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LogicLayer(nn.Module):
    def __init__(self, num_codes: int, predicates: List[str], constraints: List[Dict[str, str]]) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.predicates = predicates
        self.constraints = constraints
        self.predicate_nets = nn.ModuleDict({
            name: PredicateNet(num_codes) for name in predicates
        })

    def forward(self, codes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # codes: [B, K] integers
        b, k = codes.shape
        one_hot = torch.nn.functional.one_hot(codes, self.num_codes).float()
        preds = {name: self.predicate_nets[name](one_hot) for name in self.predicates}

        losses = []
        for constraint in self.constraints:
            ctype = constraint.get("type")
            weight = float(constraint.get("weight", 1.0))
            if ctype == "forall":
                pred = preds[constraint["pred"]]
                sat = pred.mean(dim=1)
            elif ctype == "exists":
                pred = preds[constraint["pred"]]
                sat = pred.max(dim=1).values
            elif ctype == "implies":
                p = preds[constraint["p"]]
                q = preds[constraint["q"]]
                sat = (1.0 - p + p * q).mean(dim=1)
            else:
                continue
            losses.append(weight * (1.0 - sat).mean())
        loss = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=codes.device)
        return {"logic_loss": loss, "predicate_scores": preds}
