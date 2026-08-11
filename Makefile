.PHONY: help dev infra infra-down infra-logs clean sync-deps run

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

run:
	@uv run python main.py

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