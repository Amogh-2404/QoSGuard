.PHONY: help build up down train test clean

# QoSGuard Development Commands

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	@echo "Building QoSGuard Docker images..."
	docker-compose build

up: ## Start all services
	@echo "Starting QoSGuard services..."
	docker-compose up -d
	@echo "Services started!"
	@echo "Dashboard: http://localhost:3000"
	@echo "API docs: http://localhost:8000/docs"
	@echo "MLflow: http://localhost:5000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001 (admin/admin)"

down: ## Stop all services
	@echo "Stopping QoSGuard services..."
	docker-compose down

logs: ## View logs from all services
	docker-compose logs -f

train: ## Train ML models
	@echo "Training QoSGuard models..."
	@if [ ! -d "models/registry" ]; then mkdir -p models/registry; fi
	python models/training/train_models.py

test: ## Run tests
	@echo "Running tests..."
	python -m pytest tests/ -v

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	cd ui && npm install

# Development shortcuts
api-logs: ## View API logs
	docker-compose logs -f api

ui-logs: ## View UI logs  
	docker-compose logs -f ui

restart-api: ## Restart API service
	docker-compose restart api

restart-ui: ## Restart UI service
	docker-compose restart ui
