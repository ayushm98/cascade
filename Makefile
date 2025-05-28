.PHONY: help install dev test lint format run run-ui docker-build docker-up docker-down clean

# Default target
help:
	@echo "Cascade - Intelligent LLM Router"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies with Poetry"
	@echo "  dev         Install with dev dependencies"
	@echo "  test        Run tests with pytest"
	@echo "  lint        Run linting (ruff + black check)"
	@echo "  format      Format code with black"
	@echo ""
	@echo "Running:"
	@echo "  run         Start the API server"
	@echo "  run-ui      Start the Streamlit UI"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start all services with docker-compose"
	@echo "  docker-down   Stop all services"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-railway  Deploy to Railway"
	@echo "  deploy-fly      Deploy to Fly.io"
	@echo ""
	@echo "Other:"
	@echo "  clean       Clean up build artifacts"

# Development
install:
	poetry install --no-dev

dev:
	poetry install

test:
	PYTHONPATH=src poetry run pytest tests/ -v --cov=src/cascade

lint:
	poetry run ruff check src/
	poetry run black --check src/

format:
	poetry run black src/
	poetry run ruff check --fix src/

# Running locally
run:
	PYTHONPATH=src poetry run uvicorn cascade.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	PYTHONPATH=src poetry run streamlit run src/cascade/ui/app.py --server.port 8501

# Docker
docker-build:
	docker build -t cascade:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Deployment
deploy-railway:
	@echo "Deploying to Railway..."
	@echo "1. Install Railway CLI: npm install -g @railway/cli"
	@echo "2. Login: railway login"
	@echo "3. Initialize: railway init"
	@echo "4. Deploy: railway up"
	railway up

deploy-fly:
	@echo "Deploying to Fly.io..."
	@echo "1. Install Fly CLI: curl -L https://fly.io/install.sh | sh"
	@echo "2. Login: fly auth login"
	@echo "3. Launch: fly launch"
	@echo "4. Deploy: fly deploy"
	fly deploy

# ML Training
train:
	PYTHONPATH=src poetry run python -m ml.training.train --dataset easy2hard --epochs 5

export-onnx:
	PYTHONPATH=src poetry run python -m ml.export.convert_to_onnx

# Cleanup
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
