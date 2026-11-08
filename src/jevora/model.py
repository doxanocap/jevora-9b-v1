"""Public inference-only loader for verified Jevora release packages."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import string
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .adaptation import install_adaptation
from .contracts import (
    InvalidInput, InputCapacityError,
    CHECKPOINT_FORMAT, PROMPT_NATIVE_CHAT, PROMPT_V1, PROMPT_V2,
    RELEASE_FORMAT, SAFETENSORS_INDEX, SAFETENSORS_WEIGHT_FORMAT,
    SHARDED_RELEASE_FORMAT,
)
from .package import (
    checkpoint_files, reject_absolute_paths, reject_symlinks, release_files,
    sha256, shared_tensor_aliases, verify_checkpoint, verify_release_package,
    verify_safetensors_layout,
)


def candidate_codes() -> list[str]:
    pairs = itertools.product(string.ascii_uppercase, repeat=2)
    return list(string.ascii_uppercase) + ["".join(pair) for pair in pairs]


def answer_vocabulary(tokenizer: Any, max_options: int = 255) -> tuple[list[str], list[int]]:
    cached = getattr(tokenizer, "_jevora_answer_vocabulary", None)
    if cached is not None and len(cached[0]) >= max_options:
        return cached[0][:max_options], cached[1][:max_options]
    prefix = "Answer:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    codes, token_ids = [], []
    for code in candidate_codes():
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) == 1 and tokenizer.encode(prefix + code, add_special_tokens=False) == prefix_ids + encoded:
            codes.append(code)
            token_ids.append(encoded[0])
        if len(codes) == max_options:
            break
    if len(codes) != max_options or len(set(token_ids)) != max_options:
        raise ValueError(f"tokenizer must provide {max_options} distinct single-token answer codes after {prefix!r}")
    tokenizer._jevora_answer_vocabulary = (codes, token_ids)
    return codes, token_ids


def prompt(context: str, options: list[str], prompt_version: str = PROMPT_V1, codes: list[str] | None = None) -> str:
    if prompt_version == PROMPT_V1:
        lines = ["Choose the best option for this request.", "", "Request:", context, "", "Options:"]
        lines.extend(f"{index}: {option}" for index, option in enumerate(options))
        return "\n".join(lines)
    if prompt_version != PROMPT_V2:
        raise ValueError(f"unsupported prompt version: {prompt_version}")
    if codes is None or len(codes) != len(options):
        raise ValueError("LM-head prompt requires one answer code per option")
    lines = ["State:", context, "", "Question:", "Choose the best matching option.", "", "Options:"]
    lines.extend(f"{code}: {option}" for code, option in zip(codes, options, strict=True))
    lines.extend(["", "Return only the letter code of the best option.", "", "Answer:"])
    return "\n".join(lines)


def encode(tokenizer: Any, context: str, options: list[str], max_length: int, device: torch.device, prompt_version: str) -> dict[str, torch.Tensor]:
    if not isinstance(context, str) or not context.strip():
        raise InvalidInput("context must be a non-empty string")
    if not isinstance(options, list):
        raise InvalidInput("options must be an array")
    if not 2 <= len(options) <= 255:
        raise InputCapacityError("options must contain between 2 and 255 non-empty strings")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise InvalidInput("options must contain non-empty strings")

    codes = answer_vocabulary(tokenizer, len(options))[0] if prompt_version in {PROMPT_V2, PROMPT_NATIVE_CHAT} else None
    if prompt_version == PROMPT_NATIVE_CHAT:
        messages = [
            {"role": "system", "content": "Choose the best matching candidate for the request. Return only its letter code."},
            {"role": "user", "content": json.dumps({
                "request": context,
                "candidates": [{"code": code, "answer": option} for code, option in zip(codes, options, strict=True)],
            }, ensure_ascii=False, separators=(",", ":"))},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:"
        batch = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
    else:
        batch = tokenizer(prompt(context, options, prompt_version, codes), return_tensors="pt", truncation=False)
    if batch["input_ids"].shape[1] > max_length:
        raise InputCapacityError(
            f"input has {batch['input_ids'].shape[1]} tokens, exceeds max_length={max_length}"
        )
    return {name: value.to(device) for name, value in batch.items()}


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, max_options: int = 255, answer_token_ids: list[int] | None = None) -> None:
        super().__init__()
        self.backbone, self.max_options = backbone, max_options
        dtype = next(backbone.parameters()).dtype
        self.head = nn.Linear(backbone.config.hidden_size, max_options, bias=False, dtype=dtype)
        if answer_token_ids is not None:
            output = backbone.get_output_embeddings()
            if output is None or len(answer_token_ids) != max_options or output.weight.shape[1] != backbone.config.hidden_size:
                raise ValueError("backbone does not expose a compatible LM head")
            with torch.no_grad():
                self.head.weight.copy_(output.weight[answer_token_ids].to(dtype=dtype))

    def forward(self, batch: dict[str, torch.Tensor], option_count: int) -> torch.Tensor:
        if not 2 <= option_count <= self.max_options:
            raise ValueError(f"option_count must be in [2, {self.max_options}]")
        output = self.backbone.model(**batch, use_cache=False)
        positions = batch["attention_mask"].sum(dim=1).clamp_min(1) - 1
        state = output.last_hidden_state[torch.arange(batch["input_ids"].shape[0], device=positions.device), positions]
        return self.head(state)[:, :option_count]


def decision(
    model: DecisionModel,
    tokenizer: Any,
    context: str,
    options: list[str],
    max_length: int,
    device: torch.device,
    prompt_version: str,
    temperature: float,
) -> tuple[int, list[float], int]:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = encode(tokenizer, context, options, max_length, device, prompt_version)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch, len(options))[0].float() / temperature, dim=0)
    return (
        int(probabilities.argmax()),
        [float(value) for value in probabilities.cpu()],
        int(batch["input_ids"].shape[1]),
    )


def validate_checkpoint_metadata(metadata: dict[str, Any]) -> None:
    if metadata.get("scorer_type", "index_head") != "index_head":
        raise ValueError("public loader supports only index_head checkpoints")
    if metadata.get("prompt_version", PROMPT_V1) not in {PROMPT_V1, PROMPT_V2, PROMPT_NATIVE_CHAT}:
        raise ValueError("public loader does not support this checkpoint prompt version")
    if "projection_parameter_dtype" in (metadata.get("adaptation") or {}):
        raise ValueError("public loader does not yet support the separate FP32 projection precision arm")


def load_checkpoint(directory: Path, device: torch.device, verification: dict[str, Any] | None = None) -> tuple[DecisionModel, Any, dict[str, Any]]:
    if verification is not None and verification.get("verified") is not True:
        raise ValueError(f"checkpoint integrity verification is required: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    validate_checkpoint_metadata(metadata)
    dtype = getattr(torch, metadata["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
    prompt_version = metadata.get("prompt_version", PROMPT_V1)
    answer_token_ids = None
    if prompt_version in {PROMPT_V2, PROMPT_NATIVE_CHAT}:
        codes, answer_token_ids = answer_vocabulary(tokenizer, metadata["max_options"])
        if metadata.get("answer_codes") != codes or metadata.get("answer_token_ids") != answer_token_ids:
            raise ValueError("checkpoint answer vocabulary differs from its tokenizer")
    backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(directory / "backbone_config"))
    model = DecisionModel(backbone, answer_token_ids=answer_token_ids).to(dtype=dtype)
    install_adaptation(model, metadata.get("adaptation"))
    weight_format = metadata.get("weight_format", "pytorch-state-dict")
    if weight_format == "pytorch-state-dict":
        state_dict = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file

        index = json.loads((directory / metadata.get("weight_index", SAFETENSORS_INDEX)).read_text(encoding="utf-8"))
        state_dict = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            shard_state = load_file(str(directory / shard_name), device="cpu")
            if state_dict.keys() & shard_state.keys():
                raise ValueError("checkpoint shards repeat tensor keys")
            state_dict.update(shard_state)
        aliases = index["metadata"].get("shared_tensors", {})
        if aliases != shared_tensor_aliases(model.state_dict()):
            raise ValueError("checkpoint shared-tensor aliases differ from loaded architecture")
        for alias, canonical in aliases.items():
            if canonical not in state_dict or alias in state_dict:
                raise ValueError(f"checkpoint shared-tensor alias is invalid: {alias} -> {canonical}")
            state_dict[alias] = state_dict[canonical]
        if state_dict.keys() != set(model.state_dict()):
            raise ValueError("checkpoint safetensors keys differ from model architecture")
    model.load_state_dict(state_dict)
    del state_dict
    return model.to(device=device).eval(), tokenizer, metadata


def load_release_package(directory: Path, device: torch.device) -> tuple[DecisionModel, Any, dict[str, Any], float]:
    package = verify_release_package(directory)
    model, tokenizer, metadata = load_checkpoint(Path(package["checkpoint"]), device, verification=package["checkpoint_verification"])
    return model, tokenizer, metadata, float(package["inference_config"]["temperature"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m jevora.model")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release_package(args.package), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
