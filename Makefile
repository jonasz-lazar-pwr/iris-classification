.PHONY: freeze install run run-analysis format lint test check clean help

# Default target
.DEFAULT_GOAL := help

# Freeze dependencies
freeze:
	pip freeze > requirements.txt

# Install dependencies
install:
	pip install -r requirements.txt

# Run the main application
run:
	python -m src.main

# Run analysis
run-analysis:
	python -m src.main_analysis

# Auto-fix & format code
format:
	ruff format src tests

# Linting (static analysis)
lint:
	ruff check src tests

# Run tests with coverage
test:
	pytest --cov=src --cov=tests --cov-report=term-missing

# Full quality check (format + lint + test)
check: format lint test

# Clean cache and artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov

# Show available commands
help:
	@echo "Available commands:"
	@echo "  make freeze	- Update requirements.txt"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run hyperparameter sweep"
	@echo "  make run-analysis - Run analysis module"
	@echo "  make format    - Auto-fix & format code"
	@echo "  make lint      - Lint code with ruff"
	@echo "  make test      - Run tests with coverage"
	@echo "  make check     - Format + lint + test"
	@echo "  make clean     - Remove cache and results"