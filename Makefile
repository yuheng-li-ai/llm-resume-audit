# llm-resume-audit — task runner
# Targets are POSIX-make compatible. Activate the conda env first:
#   conda activate llm-audit

.PHONY: help install lint format test test-fast audit-pilot audit-main clean

help:
	@echo "Available targets:"
	@echo "  install       pip install -e .[dev]"
	@echo "  lint          ruff check + black --check + mypy"
	@echo "  format        ruff --fix + black"
	@echo "  test          pytest with coverage gate (--cov-fail-under=80)"
	@echo "  test-fast     pytest -x --no-cov (quick smoke run)"
	@echo "  audit-pilot   100-cell GLM-4 Flash pilot (Phase 5.4)"
	@echo "  audit-main    5,000 x 4 main batch (Phase 5.5; OSF lock required)"
	@echo "  clean         remove caches, build artefacts, htmlcov"

install:
	pip install -e .[dev]
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	mypy src

format:
	ruff check --fix src tests
	black src tests

test:
	pytest

test-fast:
	pytest -x --no-cov

audit-pilot:
	@echo "Phase 5.4 pilot. Implement src/llm_audit/scoring/batch_runner.py first."
	python -m llm_audit.scoring.batch_runner --pilot --n 100 --model glm-4-flash

audit-main:
	@echo "Phase 5.5 main batch. Confirm OSF pre-registration is locked first."
	python -m llm_audit.scoring.batch_runner --main --n 5000

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage .coverage.*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
