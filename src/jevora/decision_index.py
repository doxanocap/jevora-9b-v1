"""Decision Index choice/noul engine for a verified portable Jevora package."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .contracts import InputCapacityError

try:
    from decision_index.engines import Engine, Unsupported
except ModuleNotFoundError as error:
    if error.name != "decision_index":
        raise

    class Unsupported(ValueError):
        """Allows direct local calls without installing the benchmark runner."""

    class Engine:
        """No runner dependency is needed for local package checks."""


def text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_context(state: Any, question: dict[str, Any]) -> str:
    instructions = question.get("instructions")
    if instructions is None:
        raise ValueError("Decision Index question is missing instructions")
    return f"State:\n{text(state)}\n\nQuestion:\n{text(instructions)}"


def choice_options(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("Decision Index choice criteria must be an object")
    keys = list(criteria)
    return keys, [f"{key}: {text(criteria[key])}" for key in keys]


class JevoraPackageEngine(Engine):
    """Uses the completed V1.6 research run's prompt and answer mapping.

    The caller binds a manifest SHA256. No mutable Hub branch or historical
    release revision is substituted for the actual local package identity.
    """

    name = "jevora-9b-v1"
    latency = "Device-synchronized request wall time including prompt construction; excludes loading."

    def __init__(self, package: str, package_manifest_sha256: str, device: str = "cuda") -> None:
        import torch
        from .model import PROMPT_V1, load_release_package

        super().__init__()
        directory = Path(package)
        if not re.fullmatch(r"[0-9a-f]{64}", package_manifest_sha256):
            raise ValueError("package_manifest_sha256 must be a concrete SHA256")
        actual = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if actual != package_manifest_sha256:
            raise ValueError("package manifest differs from the requested identity")
        self.device = torch.device(device)
        self.model, self.tokenizer, self.metadata, self.temperature = load_release_package(directory, self.device)
        self.max_options = int(self.metadata["max_options"])
        self.max_length = int(self.metadata["max_length"])
        self.model_id = f"jevora-9b-v1@sha256:{actual}"
        self.provenance = {
            "adapter": "jevora.decision_index:JevoraPackageEngine",
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "package_manifest_sha256": actual,
            "base_model": f"{self.metadata['base_model_id']}@{self.metadata['base_model_revision']}",
            "prompt_version": self.metadata.get("prompt_version", PROMPT_V1),
            "max_length": self.max_length,
            "max_options": self.max_options,
            "temperature": self.temperature,
        }

    def _decide(self, context: str, options: list[str]) -> tuple[int, list[float], int]:
        from .model import PROMPT_V1, decision

        try:
            choice, probabilities, tokens = decision(
                self.model, self.tokenizer, context, options, self.max_length,
                self.device, self.metadata.get("prompt_version", PROMPT_V1), self.temperature,
            )
        except InputCapacityError as error:
            raise Unsupported(str(error)) from error
        if len(probabilities) != len(options) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise RuntimeError("invalid Jevora probability distribution")
        if abs(sum(probabilities) - 1) > 1e-6 or not 0 <= choice < len(options):
            raise RuntimeError("invalid Jevora probability sum or choice")
        return choice, probabilities, tokens

    def __call__(self, state: Any, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        answers, traces = {}, {}
        for key, question in questions.items():
            kind = question.get("type")
            if kind == "choice":
                keys, options = choice_options(question)
                if not 2 <= len(options) <= self.max_options:
                    raise Unsupported(f"choice question {key} has {len(options)} options; Jevora supports 2..{self.max_options}")
                choice, probabilities, tokens = self._decide(render_context(state, question), options)
                answers[key] = {
                    "type": "choice",
                    "choice": keys[choice],
                    "probabilities": dict(zip(keys, probabilities, strict=True)),
                }
            elif kind == "noul":
                options = ["no", "yes"]
                _, probabilities, tokens = self._decide(render_context(state, question), options)
                answers[key] = {"type": "noul", "noul": probabilities[1]}
            else:
                raise Unsupported(f"unsupported Decision Index question type: {kind}")
            traces[key] = {"input_tokens": tokens, "options": len(options)}
        return (
            {"model": self.model_id, "answers": answers, "usage": {"input_tokens": sum(t["input_tokens"] for t in traces.values())}},
            {"questions": traces, "provenance": self.provenance},
        )

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def runtime(self) -> dict[str, Any]:
        import torch
        return {"torch": torch.__version__, "device": str(self.device), "dtype": str(next(self.model.parameters()).dtype)}

    def warmup(self) -> None:
        questions = {"warmup": {"type": "choice", "instructions": "Which color is named?", "criteria": {"red": "red", "blue": "blue"}}}
        for _ in range(2):
            self("The color is red.", questions)

    def close(self) -> None:
        pass
