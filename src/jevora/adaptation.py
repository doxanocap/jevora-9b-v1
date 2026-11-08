"""Inference-only reconstruction of self-contained Jevora LoRA checkpoints."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty((r, base.in_features), device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, r), device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        delta = F.linear(F.linear(self.dropout(inputs).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return output + (delta * self.scaling).to(output.dtype)


def install_adaptation(model: nn.Module, value: Any = None) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or value.get("format") != "jevora-adaptation-v1":
        raise ValueError("unsupported checkpoint adaptation format")
    if value.get("projection") not in {"trainable", "pretrained_fixed"} or value.get("method") not in {"full", "lora"}:
        raise ValueError("unsupported checkpoint adaptation policy")
    if value["method"] == "full":
        return
    settings = value["lora"]
    if settings["r"] < 1 or settings["alpha"] <= 0 or not 0 <= settings["dropout"] < 1:
        raise ValueError("invalid checkpoint LoRA settings")
    if settings.get("parameter_dtype") != "match_backbone":
        raise ValueError("unsupported checkpoint LoRA parameter dtype policy")
    root = model.backbone.model
    if hasattr(root, "language_model"):
        root = root.language_model
    targets = set(settings["target_modules"])
    matches = [
        (name, module)
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets
    ]
    if {name.split(".")[-1] for name, _ in matches} != targets:
        raise ValueError("checkpoint LoRA target coverage differs")

    for name, module in matches:
        parent_name, _, leaf = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, leaf, LoRALinear(module, settings["r"], settings["alpha"], settings["dropout"]))
