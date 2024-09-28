.PHONY: train

DOCKER_COMPOSE = sudo docker-compose -f
DOCKER_DOWN = down
DOCKER_BUILD = build
DOCKER_UP = up -d
DOCKER_LOGS = logs -f

default: train

train:
	$(DOCKER_COMPOSE) ./docker-compose.yml -p train $(DOCKER_DOWN)
	$(DOCKER_COMPOSE) ./docker-compose.yml -p train $(DOCKER_BUILD) # --no-cache
	$(DOCKER_COMPOSE) ./docker-compose.yml -p train $(DOCKER_UP) # --build
	$(DOCKER_COMPOSE) ./docker-compose.yml -p train $(DOCKER_LOGS) train

