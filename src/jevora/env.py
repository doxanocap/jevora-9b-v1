"""Pass an optional Hugging Face token from .env to a child process."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def values_from_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "HF_TOKEN":
            value = value.strip()
            values["HF_TOKEN"] = value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'} else value.split(" #", maxsplit=1)[0].strip()
    return values


def main() -> None:
    for name, value in values_from_env_file(Path(os.environ.get("JEVORA_ENV_FILE", ".env"))).items():
        if name not in os.environ:
            os.environ[name] = value
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("expected a command after --")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
