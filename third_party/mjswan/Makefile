.PHONY: sync
sync:
	uv sync --all-extras

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-all
test-all: check test

.PHONY: docs-build
docs-build:
	uv run --with-requirements docs/requirements.txt mkdocs build -f docs/mkdocs.yml

.PHONY: docs-serve
docs-serve:
	uv run --with-requirements docs/requirements.txt mkdocs serve -f docs/mkdocs.yml
