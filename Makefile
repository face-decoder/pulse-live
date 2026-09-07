.PHONY: help dev infra infra-down infra-logs clean sync-deps sync-deps-cuda run run-cuda build-opencv-cuda

OPENCV_CUDA_ROOT ?= $(HOME)/dev-build/opencv_install
CUDA_ROOT ?= /usr/local/cuda

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

run:
	@uv run python main.py

run-cuda: ## Run with the cuda extra active (plain `uv run` would silently resync back to the CPU wheel)
	@LD_LIBRARY_PATH="$(OPENCV_CUDA_ROOT)/lib:$(OPENCV_CUDA_ROOT)/lib64:$(CUDA_ROOT)/targets/x86_64-linux/lib:$$LD_LIBRARY_PATH" uv run --extra cuda python main.py

build-opencv-cuda: ## Build OpenCV+CUDA from source and package it into packages/ (long-running)
	@./scripts/build-opencv-cuda.sh

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

sync-deps: ## Sync deps, CPU-only
	@uv sync

sync-deps-cuda: ## Sync deps incl. CUDA extras
	@uv sync --extra cuda
