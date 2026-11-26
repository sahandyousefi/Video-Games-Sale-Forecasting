PYTHON ?= python

MODEL ?= lightgbm

.PHONY: install install-all lint format test check evaluate train series tune clean

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

series:
	$(PYTHON) -m vgsales series

tune:
	$(PYTHON) -m vgsales tune --model $(MODEL)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
