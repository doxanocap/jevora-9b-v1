"""Serve decisions from one verified jevora-release-v1 package."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .model import DecisionModel, decision, load_release_package


class DecisionService:
    def __init__(self, model: DecisionModel, tokenizer: Any, max_length: int, device: torch.device, prompt_version: str, temperature: float) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.max_length, self.device = max_length, device
        self.prompt_version, self.temperature = prompt_version, temperature

    def decide(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        context, options = payload.get("context"), payload.get("options")
        if not isinstance(context, str) or not isinstance(options, list):
            raise ValueError("context must be a string and options must be an array")
        choice, probabilities, _ = decision(self.model, self.tokenizer, context, options, self.max_length, self.device, self.prompt_version, self.temperature)
        return {"choice": choice, "probabilities": probabilities}


def handler(service: DecisionService) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/decision":
                self.respond(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                self.respond(HTTPStatus.OK, service.decide(payload))
            except (ValueError, json.JSONDecodeError) as error:
                self.respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except RuntimeError as error:
                self.respond(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def do_GET(self) -> None:
            self.respond(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def respond(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(prog="jevora-serve")
    parser.add_argument("--package", type=Path, required=True, help="unpacked verified jevora-release-v1 package")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, tokenizer, metadata, temperature = load_release_package(args.package, device)
    server = ThreadingHTTPServer((args.host, args.port), handler(DecisionService(model, tokenizer, metadata["max_length"], device, metadata.get("prompt_version", "jevora-v0.1-choice"), temperature)))
    print(f"listening on http://{args.host}:{args.port}/decision")
    server.serve_forever()


if __name__ == "__main__":
    main()
