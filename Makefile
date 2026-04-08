.PHONY: up down logs restart build clean help

up:           ## Start all services (mongo, qdrant, redis, minio, postgres, api)
	docker compose up -d

up-all:       ## Start everything including Streamlit
	docker compose --profile testing up -d

down:         ## Stop all services
	docker compose down

down-clean:   ## Stop and remove all data volumes
	docker compose down -v

logs:         ## Watch API logs
	docker compose logs -f api

build:        ## Rebuild images from scratch
	docker compose build --no-cache

shell:        ## Open bash in API container
	docker compose exec api bash

mongo-shell:  ## Open MongoDB shell
	docker compose exec mongo mongosh -u admin -p changeme --authenticationDatabase admin

qdrant-dash:  ## Open Qdrant dashboard URL
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"

redis-cli:    ## Open Redis CLI
	docker compose exec redis redis-cli

minio-console: ## Open MinIO console URL
	@echo "MinIO console: http://localhost:9001 (minioadmin/minioadmin)"

pg-shell:     ## Open PostgreSQL shell
	docker compose exec postgres psql -U postgres -d judge_assistant

help:         ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
