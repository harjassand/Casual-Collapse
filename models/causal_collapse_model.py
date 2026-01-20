from typing import Dict, Optional, Tuple

import torch
from torch import nn

from models.slot_encoder import SlotEncoder
from models.quantizer_vq import VectorQuantizer
from models.graph_infer import GraphInfer
from models.dynamics import SlotDynamics
from models.decoder import SlotDecoder
from models.logic_layer import LogicLayer


class CausalCollapseModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        input_type = cfg["input_type"]
        self.slot_encoder = SlotEncoder(
            input_type=input_type,
            in_dim=cfg["obs_dim"],
            slot_dim=cfg["slot_dim"],
            num_slots=cfg["num_slots"],
            image_channels=cfg.get("image_channels", 3),
            recon=cfg.get("recon", False),
        )
        self.use_quantizer = cfg.get("use_quantizer", True)
        self.quantizer = None
        if self.use_quantizer:
            self.quantizer = VectorQuantizer(
                num_codes=cfg["num_codes"],
                dim=cfg["slot_dim"],
                commitment_weight=cfg["commitment_weight"],
                decay=cfg.get("vq_decay", 0.99),
                use_ema=cfg.get("vq_use_ema", True),
            )
        self.graph_infer = GraphInfer(cfg["slot_dim"], cfg.get("graph_hidden", 64)) if cfg.get("use_graph", False) else None
        self.dynamics = SlotDynamics(
            slot_dim=cfg["slot_dim"] + (cfg["residual_dim"] if cfg.get("use_residual", False) else 0),
            action_dim=cfg.get("action_dim", 0),
            hidden_dim=cfg.get("dyn_hidden", 128),
            use_graph=cfg.get("use_graph", False),
        )
        self.decoder = SlotDecoder(
            slot_dim=cfg["slot_dim"] + (cfg["residual_dim"] if cfg.get("use_residual", False) else 0),
            out_dim=cfg["obs_dim"],
        )
        self.logic_layer = None
        if cfg.get("use_logic", False):
            self.logic_layer = LogicLayer(
                num_codes=cfg["num_codes"],
                predicates=cfg.get("logic_predicates", []),
                constraints=cfg.get("logic_constraints", []),
            )

    def encode(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        slots, attn, recon = self.slot_encoder(obs)
        if self.use_quantizer and self.quantizer is not None:
            quantized, vq_loss, vq_stats = self.quantizer(slots)
        else:
            quantized = slots
            vq_loss = torch.tensor(0.0, device=slots.device)
            indices = torch.zeros(slots.shape[0], slots.shape[1], dtype=torch.long, device=slots.device)
            usage = torch.ones(self.cfg["num_codes"], device=slots.device) / self.cfg["num_codes"]
            vq_stats = {
                "indices": indices,
                "codebook_loss": torch.tensor(0.0, device=slots.device),
                "commitment_loss": torch.tensor(0.0, device=slots.device),
                "perplexity": torch.tensor(float(self.cfg["num_codes"]), device=slots.device),
                "usage": usage,
            }
        residual = slots - quantized
        return {
            "slots": slots,
            "quantized": quantized,
            "codes": vq_stats["indices"],
            "vq_loss": vq_loss,
            "vq_stats": vq_stats,
            "attn": attn,
            "recon": recon,
            "residual": residual,
        }

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor], horizon: int) -> Dict[str, torch.Tensor]:
        enc = self.encode(obs)
        z = enc["quantized"]
        if self.cfg.get("use_residual", False):
            z = torch.cat([z, enc["residual"]], dim=-1)
        adj = self.graph_infer(enc["quantized"]) if self.graph_infer is not None else None
        rollout = self.dynamics.rollout(z, actions, adj, horizon)
        preds = self.decoder(rollout)
        out = {"preds": preds, "adj": adj}
        out.update(enc)
        return out
