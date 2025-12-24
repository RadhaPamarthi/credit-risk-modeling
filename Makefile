# Environment: dev (default) or prod
ENV ?= dev
PYTHON ?= python

.PHONY: setup train score test clean

# Create virtual environment and install dependencies
setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Train model using Metaflow pipeline
train:
	ENV=$(ENV) $(PYTHON) flow/credit_risk_flow.py run --data_path data/credit_risk_data_enhanced.csv

# Run production scoring
# Usage: make score INPUT=data.csv OUTPUT=scores.parquet
score:
	ENV=$(ENV) $(PYTHON) scripts/score_production.py \
		--config config/config.yaml \
		--input $(INPUT) \
		--output $(OUTPUT)

# Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v

# Run linting
lint:
	$(PYTHON) -m flake8 src/ --max-line-length=100

# Remove generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .metaflow htmlcov .coverage