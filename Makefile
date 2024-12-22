ENV_VARS ?= PYTHONNOUSERSITE=1 \
        PYTHONPATH=${PYTHONPATH}:${PWD}

build: Dockerfile
	DOCKER_BUILDKIT=1 docker build --progress=plain  -t pytorch_gpu:0.1 . 

env: build
	docker run -it  --gpus all --net=host  -v ~/.bashrc:/root/.bashrc -v ~/.gitconfig:/root/.gitconfig -v ${PWD}:/app/ -v ~/:/home/user/ -w /app/ pytorch_gpu:0.1 /bin/bash

jupyter:
	echo ${ENV_VARS}
	env ${ENV_VARS} jupyter notebook --allow-root --port 8890 --NotebookApp.token='' --NotebookApp.password=''
	