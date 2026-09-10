# Convenience targets. If `make` is not installed (default on Windows), run the
# underlying commands directly — they are shown in each recipe.
#
# On Windows: `winget install GnuWin32.Make` or just use the raw commands.

PY ?= .venv/Scripts/python

.PHONY: help install install-app install-ingest install-dev lock lint format hooks ingest run test eval

help:
	@echo "install         install everything (ingestion + app)"
	@echo "install-app     install only what the Streamlit app needs"
	@echo "install-ingest  install only what the ingestion pipeline needs"
	@echo "install-dev     install app + dev tooling (ruff, pytest, pre-commit)"
	@echo "lock            recompile requirements.lock from requirements.txt (needs uv)"
	@echo "lint            ruff check + format --check"
	@echo "format          ruff check --fix + format"
	@echo "hooks           install and run the pre-commit hooks"
	@echo "ingest          build the FAISS index from data/document.pdf"
	@echo "run             start the Streamlit app"
	@echo "test            run the test suite"
	@echo "eval            run the retrieval evaluation harness"

install:
	$(PY) -m pip install -r requirements.txt

install-app:
	$(PY) -m pip install -r requirements-app.txt

install-ingest:
	$(PY) -m pip install -r requirements-ingest.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

lock:
	uv pip compile requirements.txt -o requirements.lock

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

hooks:
	$(PY) -m pre_commit install
	$(PY) -m pre_commit run --all-files

ingest:
	$(PY) ingest.py

run:
	$(PY) -m streamlit run app.py

test:
	$(PY) -m pytest -q

eval:
	$(PY) eval/run_eval.py
