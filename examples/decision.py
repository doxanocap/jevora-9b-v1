"""Minimal local use of an unpacked, verified jevora-9b-v1 model package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jevora.model import decision, load_release_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    model, tokenizer, metadata, temperature = load_release_package(args.package, device)
    context = "A customer needs to pay at a shop using their bank card."
    options = ["card payment", "cash withdrawal", "bank transfer"]
    choice, probabilities, tokens = decision(
        model, tokenizer, context, options, metadata["max_length"],
        device, metadata.get("prompt_version", "jevora-v0.1-choice"), temperature,
    )
    result = {
        "context": context,
        "options": options,
        "choice": choice,
        "probabilities": probabilities,
        "rendered_tokens": tokens,
        "temperature": temperature,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
