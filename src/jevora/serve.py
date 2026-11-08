"""Serve decisions from one verified jevora-release-v1 package."""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from .contracts import InvalidInput
from .model import DecisionModel, decision, load_release_package


class DecisionService:
    def __init__(
        self,
        model: DecisionModel,
        tokenizer: Any,
        max_length: int,
        device: torch.device,
        prompt_version: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.prompt_version = prompt_version
        self.temperature = temperature

    def decide(self, context: str, options: list[str]) -> dict[str, Any]:
        choice, probabilities, _ = decision(
            self.model, self.tokenizer, context, options, self.max_length,
            self.device, self.prompt_version, self.temperature,
        )
        return {"choice": choice, "probabilities": probabilities}


def handler(service: DecisionService) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/decision":
                self.respond(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
            except ValueError as error:
                self.respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            if not isinstance(payload, dict):
                self.respond(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "request body must be a JSON object"},
                )
                return

            try:
                result = service.decide(payload.get("context"), payload.get("options"))
            except InvalidInput as error:
                self.respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception:
                logging.exception("decision inference failed")
                self.respond(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "decision inference failed"},
                )
                return

            self.respond(HTTPStatus.OK, result)

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
    service = DecisionService(
        model=model,
        tokenizer=tokenizer,
        max_length=metadata["max_length"],
        device=device,
        prompt_version=metadata.get("prompt_version", "jevora-v0.1-choice"),
        temperature=temperature,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler(service))
    try:
        print(f"listening on http://{args.host}:{args.port}/decision")
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
