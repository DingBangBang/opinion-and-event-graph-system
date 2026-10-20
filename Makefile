SHELL := /bin/bash
PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: help install install-ml test demo api producer up down logs ps clean

help:
	@echo "install     : pip install core dependencies (activated conda env)"
	@echo "install-ml  : install torch/transformers for HF backends"
	@echo "test        : run pytest"
	@echo "demo        : offline end-to-end demo + figures"
	@echo "api         : run FastAPI locally"
	@echo "producer    : send mock comments to localhost:29092"
	@echo "up          : docker compose up -d --build (infra + api + processor)"
	@echo "up-demo     : docker compose --profile demo up -d --build"
	@echo "down        : stop containers (keeps volumes)"
	@echo "clean       : remove generated demo data/figures"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-ml:
	$(PYTHON) -m pip install -r requirements-ml.txt

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/seed_demo_data.py

api:
	uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

producer:
	$(PYTHON) scripts/mock_producer.py --count 200 --rate 5

up:
	$(COMPOSE) up -d --build

up-demo:
	$(COMPOSE) --profile demo up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f processor api

ps:
	$(COMPOSE) ps

clean:
	rm -f data/processed.jsonl data/routing_audit.jsonl data/decisions.jsonl \
		data/alerts.jsonl data/graph_snapshot.json data/demo_summary.json \
		data/sentiment_price.csv
	rm -f reports/*.png reports/demo_metrics.json
