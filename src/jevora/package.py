"""Integrity checks for portable Jevora model packages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    CHECKPOINT_FORMAT, PROMPT_V1, RELEASE_FORMAT, SAFETENSORS_INDEX,
    SAFETENSORS_WEIGHT_FORMAT, SHARDED_RELEASE_FORMAT,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "integrity.json"
    }


def release_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


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


def verify_safetensors_layout(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    index_name = metadata.get("weight_index", SAFETENSORS_INDEX)
    index_path = Path(index_name)
    if index_path.is_absolute() or ".." in index_path.parts or len(index_path.parts) != 1:
        raise ValueError("checkpoint weight index must be a filename inside the checkpoint")
    index_path = directory / index_path
    if not index_path.is_file():
        raise ValueError(f"checkpoint safetensors index is missing: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_metadata = payload.get("metadata")
    weight_map = payload.get("weight_map")
    aliases = index_metadata.get("shared_tensors", {}) if isinstance(index_metadata, dict) else None
    if (
        not isinstance(index_metadata, dict)
        or index_metadata.get("jevora_format") != SAFETENSORS_WEIGHT_FORMAT
        or index_metadata.get("format") != "pt"
        or not isinstance(weight_map, dict)
        or not weight_map
        or not isinstance(aliases, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items())
    ):
        raise ValueError(f"checkpoint safetensors index is malformed: {index_path}")
    shard_names = set()
    for tensor_name, filename in weight_map.items():
        relative = Path(filename) if isinstance(filename, str) else Path("../invalid")
        if (
            not isinstance(tensor_name, str)
            or not tensor_name
            or relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 1
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(f"checkpoint index contains an unsafe tensor or shard name: {tensor_name!r}")
        shard_names.add(filename)
    shard_paths = {path.name for path in directory.glob("*.safetensors") if path.is_file()}
    if shard_names != shard_paths:
        raise ValueError("checkpoint safetensors index and shard files differ")
    if any(alias in weight_map or target not in weight_map or alias == target for alias, target in aliases.items()):
        raise ValueError("checkpoint shared-tensor aliases do not point to stored canonical tensors")
    if not isinstance(index_metadata.get("total_size"), int) or index_metadata["total_size"] < 0:
        raise ValueError("checkpoint safetensors index has invalid total_size")
    if not isinstance(index_metadata.get("max_shard_size_bytes"), int) or index_metadata["max_shard_size_bytes"] <= 0:
        raise ValueError("checkpoint safetensors index has invalid max_shard_size_bytes")
    from safetensors import safe_open

    available_names: set[str] = set()
    for filename in sorted(shard_names):
        with safe_open(directory / filename, framework="pt", device="cpu") as shard:
            names = set(shard.keys())
        if available_names.intersection(names):
            raise ValueError(f"checkpoint safetensors shards contain duplicate tensor keys: {filename}")
        available_names.update(names)
        if any(weight_map[name] != filename for name in names if name in weight_map):
            raise ValueError(f"checkpoint safetensors shard disagrees with index: {filename}")
    if available_names != set(weight_map):
        raise ValueError("checkpoint safetensors index tensor keys differ from shard contents")
    return payload


def verify_checkpoint(directory: Path, files_sha256: dict[str, str] | None = None) -> dict[str, Any]:
    required = ("metadata.json", "backbone_config/config.json", "tokenizer/tokenizer_config.json", "integrity.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is incomplete: {directory}; missing: {', '.join(missing)}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    weight_format = metadata.get("weight_format", "pytorch-state-dict")
    if weight_format == "pytorch-state-dict":
        if not (directory / "model.pt").is_file():
            raise ValueError(f"checkpoint is incomplete: {directory}; missing: model.pt")
        if (directory / SAFETENSORS_INDEX).exists():
            raise ValueError("checkpoint has ambiguous PyTorch and safetensors weight files")
    elif weight_format == SAFETENSORS_WEIGHT_FORMAT:
        verify_safetensors_layout(directory, metadata)
        if (directory / "model.pt").exists():
            raise ValueError("sharded safetensors checkpoint must not also contain model.pt")
    else:
        raise ValueError(f"checkpoint has unsupported weight_format: {weight_format}")
    integrity = json.loads((directory / "integrity.json").read_text(encoding="utf-8"))
    if integrity.get("format") != CHECKPOINT_FORMAT or integrity.get("complete") is not True:
        raise ValueError(f"checkpoint is not complete: {directory / 'integrity.json'}")
    actual = files_sha256 if files_sha256 is not None else checkpoint_files(directory)
    if integrity.get("files_sha256") != actual:
        raise ValueError(f"checkpoint file checksums differ: {directory}")
    return {
        "format": CHECKPOINT_FORMAT,
        "weight_format": weight_format,
        "verified": True,
        "files_sha256": actual,
    }


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
    package_format = manifest.get("format")
    if not isinstance(model_name, str) or not model_name.strip() or package_format not in (RELEASE_FORMAT, SHARDED_RELEASE_FORMAT) or manifest.get("complete") is not True:
        raise ValueError(f"release package is not complete: {directory / 'manifest.json'}")
    actual_files = release_files(directory)
    if manifest.get("files_sha256") != actual_files:
        raise ValueError(f"release package file checksums differ: {directory}")
    checkpoint = directory / "checkpoint"
    if (checkpoint / "training_state.pt").exists():
        raise ValueError(f"release package contains forbidden optimizer state: {checkpoint / 'training_state.pt'}")
    checkpoint_file_hashes = {
        name.removeprefix("checkpoint/"): digest
        for name, digest in actual_files.items()
        if name.startswith("checkpoint/") and name != "checkpoint/integrity.json"
    }
    verification = verify_checkpoint(checkpoint, files_sha256=checkpoint_file_hashes)
    if manifest.get("checkpoint_files_sha256") != verification["files_sha256"]:
        raise ValueError(f"release package checkpoint checksums differ: {directory}")
    inference = verify_inference_policy(directory, manifest, verification)
    return {
        "format": package_format,
        "model_name": model_name,
        "package": str(directory),
        "checkpoint": str(checkpoint),
        "inference_config": inference,
        "files_sha256": actual_files,
        "checkpoint_verification": verification,
    }


def verify_inference_policy(
    directory: Path,
    manifest: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Check that the packaged inference policy belongs to its weights."""
    model_name = manifest["model_name"]
    package_format = manifest["format"]
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("format") != "jevora-release-calibration-v1" or calibration.get("checkpoint_files_sha256") != verification["files_sha256"]:
        raise ValueError(f"release calibration does not belong to packaged checkpoint: {directory / 'calibration.json'}")
    if calibration.get("source_checkpoint_files_sha256") != manifest.get("source_checkpoint_files_sha256") or calibration.get("source_calibration_sha256") != manifest.get("source_calibration_sha256"):
        raise ValueError(f"release calibration provenance differs from manifest: {directory / 'calibration.json'}")
    inference = json.loads((directory / "inference-config.json").read_text(encoding="utf-8"))
    status = calibration.get("status", "fitted")
    if status not in {"fitted", "uncalibrated"} or inference.get("calibration_status", "fitted") != status:
        raise ValueError("release calibration status differs from inference policy")
    if status == "uncalibrated" and (
        calibration.get("temperature") != 1.0 or calibration.get("source_calibration_sha256") is not None
        or calibration.get("fit_split") is not None or calibration.get("metrics")
    ):
        raise ValueError("uncalibrated release must retain temperature=1 without fitted evidence")
    checkpoint = directory / "checkpoint"
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    reject_absolute_paths(metadata)
    export = metadata.get("portable_export", {})
    export_fields = (
        "model_name", "format", "source_checkpoint_metadata_sha256",
        "source_checkpoint_files_sha256",
    )
    if any(export.get(field) != manifest.get(field) for field in export_fields):
        raise ValueError(f"release checkpoint provenance differs from manifest: {directory}")
    expected = {
        "base_model_id": metadata["base_model_id"],
        "base_model_revision": metadata["base_model_revision"],
        "dtype": metadata["dtype"],
        "max_length": metadata["max_length"],
        "max_options": metadata["max_options"],
        "prompt_version": metadata.get("prompt_version", PROMPT_V1),
        "checkpoint_files_sha256": verification["files_sha256"],
        "model_name": model_name,
    }
    if package_format == SHARDED_RELEASE_FORMAT:
        expected["weight_format"] = SAFETENSORS_WEIGHT_FORMAT
    elif metadata.get("weight_format", "pytorch-state-dict") != "pytorch-state-dict":
        raise ValueError("legacy release format must use model.pt")
    elif "weight_format" in inference:
        expected["weight_format"] = "pytorch-state-dict"
    expected["format"] = package_format
    mismatched = [key for key, value in expected.items() if inference.get(key) != value]
    if mismatched:
        raise ValueError(f"release inference config differs from package: {', '.join(mismatched)}")

    temperature = inference.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
        or calibration.get("temperature") != temperature
    ):
        raise ValueError("release inference temperature differs from calibration")
    return inference


def shared_tensor_aliases(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    storage_names: dict[tuple[int, int], list[str]] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"unsupported checkpoint tensor: {name}")
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            continue
        storage_names.setdefault((storage.data_ptr(), storage.nbytes()), []).append(name)
    aliases: dict[str, str] = {}
    for names in storage_names.values():
        if len(names) < 2:
            continue
        canonical = names[0]
        reference = state_dict[canonical]
        signature = (reference.dtype, tuple(reference.shape), tuple(reference.stride()), reference.storage_offset())
        for name in names[1:]:
            tensor = state_dict[name]
            candidate = (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())
            if candidate != signature:
                raise ValueError(f"checkpoint has overlapping non-identical tensor views: {canonical}, {name}")
            aliases[name] = canonical
    return aliases


