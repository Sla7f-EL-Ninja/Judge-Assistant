.PHONY: up down logs restart build clean help

up:           ## Start mongo + API
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
	docker compose exec mongo mongosh

help:         ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
