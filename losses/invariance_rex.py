from typing import List

import torch


def rex_penalty(risks: List[torch.Tensor]) -> torch.Tensor:
    if len(risks) == 0:
        return torch.tensor(0.0)
    stacked = torch.stack(risks)
    return stacked.var(unbiased=False)
