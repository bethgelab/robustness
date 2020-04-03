 include config

test: docker
	sudo docker run \
		-v /home/sschneider/shared:/home/ubuntu/shared \
		-v /home/sschneider/local:/home/ubuntu/local   \
		-v /home/sschneider/phd:/home/ubuntu/phd \
		-v $(shell pwd):/home/ubuntu/run  \
		-w /home/ubuntu/run \
		--ipc host -i \
		--entrypoint ptw stes/dj . -- --failed-first --exitfirst 

simg:
	singularity build docker://bethgelab/deeplearning:cuda10.0-cudnn7-flake8  

docker:
	# Build docker container 
	sudo docker build \
	 --build-arg userid=$(shell id -u) \
	 --build-arg username=$(shell id -un) \
	 -t ${DOCKER_NAME} .

run_%:
	# Start training in interactive mode
	# see https://github.com/NVIDIA/nvidia-docker/issues/1026
	mkdir -p $(shell pwd)/.torch
	mkdir -p /home/sschneider/local/cache
	mkdir -p /home/sschneider/local/torch
	sudo docker run \
		--gpus '"device=$(GPU)"' \
		-v /home/sschneider/shared:/home/ubuntu/shared \
		-v /home/sschneider/local:/home/ubuntu/local   \
		-v /home/sschneider/phd:/home/ubuntu/phd \
		-v /mnt/qb/bethge/erusak/Better_baselines_for_ImageNet-C/models:/mnt/qb/bethge/erusak/Better_baselines_for_ImageNet-C/models \
		-v /home/sschneider/local/cache:/home/ubuntu/.cache \
		-v /home/sschneider/local/torch:/home/ubuntu/.torch \
		-v $(BASEDIR):$(BASEDIR) \
		--tmpfs /data \
		-v $(shell pwd):/home/ubuntu/run  \
		-w /home/ubuntu/run -u $(shell id -u):$(shell echo `id -G | awk '{print $$2}'`) \
		--ipc host -d \
		--entrypoint bash stes/dj ./scripts/$@.sh $(BASEDIR)

adapt:
	# Start training in interactive mode
	mkdir -p $(shell pwd)/.torch
	sudo docker run \
		--gpus device=5 \
		-v /home/sschneider/shared:/home/ubuntu/shared \
		-v /home/sschneider/local:/home/ubuntu/local   \
		-v /home/sschneider/phd:/home/ubuntu/phd \
		-v $(shell pwd)/.torch:/home/ubuntu/.cache/torch \
		-v /mnt/nvme/erusak:/data \
		-v $(shell pwd):/home/ubuntu/run  \
		-w /home/ubuntu/run -u $(shell id -u):$(shell echo `id -G | awk '{print $$2}'`) \
		--ipc host -it \
		--entrypoint python stes/dj adapt.py \
			--batch-size 256 \
			--dataset /data/ImageNet-C/brightness/5 \
			--workers 20 \
			--output test_model_2.pt \
			--verbose


format: docker
	sudo docker run \
		-v /home/sschneider/shared:/home/ubuntu/shared \
		-v /home/sschneider/local:/home/ubuntu/local   \
		-v /home/sschneider/phd:/home/ubuntu/phd \
		-v $(shell pwd):/home/ubuntu/run  \
		-w /home/ubuntu/run \
		--ipc host -i \
		--entrypoint black stes/dj *.py

stop_docker:
	sudo docker ps | grep stes/dj | awk '{print $$1}' | xargs sudo docker stop
