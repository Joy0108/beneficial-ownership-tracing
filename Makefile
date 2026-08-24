.PHONY: help install registers eval test lint clean docker deploy

PY ?= python
export PYTHONPATH := src

help:
	@echo "install    install the package with dev extras"
	@echo "registers  regenerate the six register files from the seed world"
	@echo "eval       run the full evaluation and register the version"
	@echo "test       run the test suite"
	@echo "lint       ruff check"
	@echo "clean      remove artifacts and reports"

install:
	$(PY) -m pip install -e ".[dev]"

registers:
	$(PY) scripts/build_registers.py

eval: registers
	$(PY) -m ubo.cli eval

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

clean:
	rm -rf artifacts reports .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:
	docker compose up --build

deploy:
	kubectl apply -k deploy/
