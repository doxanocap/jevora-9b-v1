.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python3
PACKAGE ?=
DEVICE ?= cpu
OUTPUT ?=
HOST ?= 127.0.0.1
PORT ?= 8080
RUN_ENV = PYTHONNOUSERSITE=1 PYTHONPATH=src JEVORA_ENV_FILE="$(CURDIR)/.env" $(PYTHON) -m jevora.env --

.PHONY: help setup doctor require-package verify-package example serve api-check clean

help:
	@printf '%s\n' \
	  'make setup' \
	  'make verify-package PACKAGE=/path/to/jevora-9b-v1' \
	  'make example PACKAGE=/path/to/jevora-9b-v1 DEVICE=cpu' \
	  'make serve PACKAGE=/path/to/jevora-9b-v1 DEVICE=cuda' \
	  'make api-check PACKAGE=/path/to/jevora-9b-v1 DEVICE=cuda'

setup:
	@$(RUN_ENV) $(UV) sync --locked
	@$(RUN_ENV) $(UV) run --no-sync python scripts/doctor.py

doctor:
	@$(RUN_ENV) $(UV) run --no-sync python scripts/doctor.py

require-package:
	@test -n "$(PACKAGE)" || { echo 'PACKAGE is required; pass the unpacked published model package' >&2; exit 2; }
	@test -d "$(PACKAGE)" || { echo "package directory is missing: $(PACKAGE)" >&2; exit 2; }

verify-package: require-package
	@$(RUN_ENV) $(UV) run --no-sync python -m jevora.model verify --package "$(PACKAGE)"

example: require-package
	@$(RUN_ENV) $(UV) run --no-sync python examples/decision.py --package "$(PACKAGE)" --device "$(DEVICE)" $(if $(OUTPUT),--output "$(OUTPUT)")

serve: require-package
	@$(RUN_ENV) $(UV) run --no-sync python -m jevora.serve --package "$(PACKAGE)" --device "$(DEVICE)" --host "$(HOST)" --port "$(PORT)"

api-check: require-package
	@$(RUN_ENV) $(UV) run --no-sync python -m jevora.api_check --package "$(PACKAGE)" --device "$(DEVICE)"

clean:
	@printf '%s\n' 'clean is intentionally disabled: this repository does not manage model packages or caches'
