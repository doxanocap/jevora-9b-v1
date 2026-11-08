"""Exercise the local /decision HTTP contract for one verified package."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch

from .model import load_release_package
from .serve import DecisionService, handler


def post(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def run(package: Path, device: str) -> dict[str, object]:
    target = torch.device(device)
    model, tokenizer, metadata, temperature = load_release_package(package, target)
    service = DecisionService(
        model=model,
        tokenizer=tokenizer,
        max_length=metadata["max_length"],
        device=target,
        prompt_version=metadata.get("prompt_version", "jevora-v0.1-choice"),
        temperature=temperature,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/decision"
        status, body = post(endpoint, {"context": "I want to pay by card", "options": ["card payment", "cash withdrawal", "bank transfer"]})
        probabilities = body.get("probabilities") if isinstance(body, dict) else None
        if status != 200 or not isinstance(probabilities, list) or len(probabilities) != 3 or abs(sum(probabilities) - 1.0) > 1e-5:
            raise RuntimeError(f"valid /decision response failed: status={status}, body={body}")
        invalid_status, _ = post(endpoint, {"context": "missing options"})
        long_status, _ = post(endpoint, {"context": "x " * (metadata["max_length"] + 20), "options": ["one", "two"]})
        if invalid_status != 400 or long_status != 400:
            raise RuntimeError(f"invalid request handling failed: invalid={invalid_status}, long={long_status}")
        return {
            "status": "passed",
            "choice": body["choice"],
            "probability_count": len(probabilities),
            "probability_sum": sum(probabilities),
            "invalid_status": invalid_status,
            "long_input_status": long_status,
            "temperature": temperature,
            "package": str(package),
        }
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="jevora-api-check")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.package, args.device), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
