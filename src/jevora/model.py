"""Public inference-only loader for one verified jevora-release-v1 package."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import string
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROMPT_V1 = "jevora-v0.1-choice"
PROMPT_V2 = "jevora-v0.2-lm-head-codes"
CHECKPOINT_FORMAT = "jevora-checkpoint-v2"
RELEASE_FORMAT = "jevora-release-v1"


def candidate_codes() -> list[str]:
    return list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]


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
        raise ValueError("context must be a non-empty string")
    if not 2 <= len(options) <= 255 or any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("options must contain between 2 and 255 non-empty strings")
    codes = answer_vocabulary(tokenizer, len(options))[0] if prompt_version == PROMPT_V2 else None
    batch = tokenizer(prompt(context, options, prompt_version, codes), return_tensors="pt", truncation=False)
    if batch["input_ids"].shape[1] > max_length:
        raise ValueError(f"input has {batch['input_ids'].shape[1]} tokens, exceeds max_length={max_length}")
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


def decision(model: DecisionModel, tokenizer: Any, context: str, options: list[str], max_length: int, device: torch.device, prompt_version: str, temperature: float) -> tuple[int, list[float], int]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = encode(tokenizer, context, options, max_length, device, prompt_version)
    with torch.inference_mode():
        probabilities = torch.softmax(model(batch, len(options))[0].float() / temperature, dim=0)
    return int(probabilities.argmax()), [float(value) for value in probabilities.cpu()], int(batch["input_ids"].shape[1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "integrity.json"}


def release_files(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}


def reject_symlinks(directory: Path, label: str) -> None:
    links = [str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"{label} must not contain symlinks: {', '.join(links[:3])}")


def reject_absolute_paths(value: Any, path: str = "metadata") -> None:
    if isinstance(value, str):
        if Path(value).is_absolute():
            raise ValueError(f"portable checkpoint contains an absolute path at {path}: {value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_absolute_paths(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_absolute_paths(item, f"{path}.{key}")


def verify_checkpoint(directory: Path) -> dict[str, Any]:
    required = ("model.pt", "metadata.json", "backbone_config/config.json", "tokenizer/tokenizer_config.json", "integrity.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is incomplete: {directory}; missing: {', '.join(missing)}")
    integrity = json.loads((directory / "integrity.json").read_text(encoding="utf-8"))
    if integrity.get("format") != CHECKPOINT_FORMAT or integrity.get("complete") is not True:
        raise ValueError(f"checkpoint is not complete: {directory / 'integrity.json'}")
    actual = checkpoint_files(directory)
    if integrity.get("files_sha256") != actual:
        raise ValueError(f"checkpoint file checksums differ: {directory}")
    return {"format": CHECKPOINT_FORMAT, "verified": True, "files_sha256": actual}


def verify_release_package(directory: Path) -> dict[str, Any]:
    if directory.is_symlink():
        raise ValueError(f"release package root must not be a symlink: {directory}")
    required = ("checkpoint", "calibration.json", "inference-config.json", "MODEL_CARD.md", "manifest.json")
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise ValueError(f"release package is incomplete: {directory}; missing: {', '.join(missing)}")
    reject_symlinks(directory, "release package")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    model_name = manifest.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip() or manifest.get("format") != RELEASE_FORMAT or manifest.get("complete") is not True:
        raise ValueError(f"release package is not complete: {directory / 'manifest.json'}")
    actual_files = release_files(directory)
    if manifest.get("files_sha256") != actual_files:
        raise ValueError(f"release package file checksums differ: {directory}")
    checkpoint = directory / "checkpoint"
    if (checkpoint / "training_state.pt").exists():
        raise ValueError(f"release package contains forbidden optimizer state: {checkpoint / 'training_state.pt'}")
    verification = verify_checkpoint(checkpoint)
    if manifest.get("checkpoint_files_sha256") != verification["files_sha256"]:
        raise ValueError(f"release package checkpoint checksums differ: {directory}")
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("format") != "jevora-release-calibration-v1" or calibration.get("checkpoint_files_sha256") != verification["files_sha256"]:
        raise ValueError(f"release calibration does not belong to packaged checkpoint: {directory / 'calibration.json'}")
    if calibration.get("source_checkpoint_files_sha256") != manifest.get("source_checkpoint_files_sha256") or calibration.get("source_calibration_sha256") != manifest.get("source_calibration_sha256"):
        raise ValueError(f"release calibration provenance differs from manifest: {directory / 'calibration.json'}")
    inference = json.loads((directory / "inference-config.json").read_text(encoding="utf-8"))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    reject_absolute_paths(metadata)
    expected = {"base_model_id": metadata["base_model_id"], "base_model_revision": metadata["base_model_revision"], "dtype": metadata["dtype"], "max_length": metadata["max_length"], "max_options": metadata["max_options"], "prompt_version": metadata.get("prompt_version", PROMPT_V1), "checkpoint_files_sha256": verification["files_sha256"], "model_name": model_name}
    mismatched = [key for key, value in expected.items() if inference.get(key) != value]
    temperature = inference.get("temperature")
    if inference.get("format") != RELEASE_FORMAT or mismatched or not isinstance(temperature, (int, float)) or temperature <= 0 or calibration.get("temperature") != temperature:
        raise ValueError(f"release inference config differs from package: {', '.join(mismatched) or 'format or temperature'}")
    return {"format": RELEASE_FORMAT, "model_name": model_name, "package": str(directory), "checkpoint": str(checkpoint), "inference_config": inference, "files_sha256": actual_files}


def load_checkpoint(directory: Path, device: torch.device) -> tuple[DecisionModel, Any, dict[str, Any]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    dtype = getattr(torch, metadata["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
    prompt_version = metadata.get("prompt_version", PROMPT_V1)
    answer_token_ids = None
    if prompt_version == PROMPT_V2:
        codes, answer_token_ids = answer_vocabulary(tokenizer, metadata["max_options"])
        if metadata.get("answer_codes") != codes or metadata.get("answer_token_ids") != answer_token_ids:
            raise ValueError("checkpoint answer vocabulary differs from its tokenizer")
    backbone = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(directory / "backbone_config"))
    model = DecisionModel(backbone, answer_token_ids=answer_token_ids).to(dtype=dtype)
    state_dict = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    return model.to(device=device).eval(), tokenizer, metadata


def load_release_package(directory: Path, device: torch.device) -> tuple[DecisionModel, Any, dict[str, Any], float]:
    package = verify_release_package(directory)
    model, tokenizer, metadata = load_checkpoint(Path(package["checkpoint"]), device)
    return model, tokenizer, metadata, float(package["inference_config"]["temperature"])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m jevora.model")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release_package(args.package), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
