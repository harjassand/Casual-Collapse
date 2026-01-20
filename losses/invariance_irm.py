from typing import Callable, List

import torch


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor, loss_fn: Callable) -> torch.Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = loss_fn(logits * scale, targets)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.pow(2)


def irm_penalty_envs(
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    loss_fn: Callable,
) -> torch.Tensor:
    penalties = []
    for logits, targets in zip(logits_list, targets_list):
        penalties.append(irm_penalty(logits, targets, loss_fn))
    return torch.stack(penalties).mean()
