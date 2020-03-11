TYPE = GPU
PROJECT_NAME = lab
IMAGE_NAME = $(USER)-$(PROJECT_NAME)-image
CONTAINER_NAME = $(USER)-$(PROJECT_NAME)-container
HOST_PORT = $(shell expr 10000 + $(shell id -u $(USER)))
NVIDIA_DOCKER_CMD = docker  # nvidia-docker is old. should use with `--gpus all`

.PHONY: build
build:
	docker build . -t $(IMAGE_NAME)

.PHONY: run
run:
	$(NVIDIA_DOCKER_CMD) run -itd --gpus all --rm --name $(CONTAINER_NAME) \
	-v $(HOST_DIR):/code \
	-p $(HOST_PORT):8888 $(IMAGE_NAME) \
	env SHELL='/bin/bash' jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/code

.PHONY: exec
exec:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Clean
.PHONY: all-clean
all-clean:
	make stop
	make container-clean
	make image-clean

.PHONY: image-clean
image-clean:
	docker rmi `docker images -q $(IMAGE_NAME)`

.PHONY: container-clean
container-clean:
	docker rm `docker ps -aq --filter name=$(CONTAINER_NAME)`

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME)

.PHONY: notebook-url
notebook-url:
	docker logs $(CONTAINER_NAME) | head
	@echo HOST_PORT: $(HOST_PORT)

.PHONY: config
config:
	@echo TYPE: $(TYPE)
	@echo IMAGE_NAME: $(IMAGE_NAME)
	@echo CONTAINER_NAME: $(CONTAINER_NAME)
	@echo HOST_DIRECTORY: $(HOST_DIR)
	@echo HOST_PORT: $(HOST_PORT)
