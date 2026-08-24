.PHONY: help dev infra infra-down infra-logs clean sync-deps run lint format test

PY := .venv/bin/python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

run:
	@uv run python main.py

# Tooling targets use the provisioned .venv directly: `uv run` re-syncs the
# environment first, which fails wherever the gitignored opencv-cuda wheel
# (packages/opencv_cuda-*.whl) is not present on disk.

lint:
	@$(PY) -m ruff check .
	@$(PY) -m ruff format --check .

format:
	@$(PY) -m ruff format .

test:
	@$(PY) -m pytest tests/ -q

infra:
	@docker compose up -d

infra-down:
	@docker compose down

infra-logs:
	@docker compose logs -f

dev:
	@uv run python main.py

clean:
	@rm -rf .tmp __pycache__ src/**/__pycache__

sync-deps:
	@./scripts/sync-deps.sh