#!/bin/bash

sudo docker pull pytorch/pytorch

cachedir=/tmp/.cache
localdir=/tmp/.local

mkdir -p $cachedir
mkdir -p $localdir

sudo docker run \
--gpus 4 --ipc host \
-v /run/user/7867/visda-2019:/data:ro \
-v $(pwd)/gen-efficientnet-pytorch:/app \
-v $cachedir:/.cache \
-v $localdir:/.local \
--user $(id -u) \
--workdir /app \
-it pytorch/pytorch ./run.sh
