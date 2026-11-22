PYTHON ?= python

.PHONY: install install-all lint format test check evaluate train clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,all]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	$(PYTHON) -m pytest

check: lint test

evaluate:
	$(PYTHON) -m vgsales evaluate

train:
	$(PYTHON) -m vgsales train

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
