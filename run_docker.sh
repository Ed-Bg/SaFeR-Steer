#!/bin/bash
# Docker environment setup for SafeDy training
# Please modify the paths according to your environment

# Configuration - modify these paths as needed
WORKSPACE_PATH="${WORKSPACE_PATH:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:-<YOUR_MODEL_PATH>}"
CONTAINER_NAME="${CONTAINER_NAME:-safedy}"

docker rm -f $CONTAINER_NAME 2>/dev/null || true
docker run -it \
  --name $CONTAINER_NAME \
  --add-host=host.docker.internal:host-gateway \
  --ipc=host \
  --gpus all \
  --network host \
  -v $WORKSPACE_PATH:/workspace \
  -v $MODEL_PATH:/workspace/models \
  -w /workspace \
  verlai/verl:sgl056.latest 

# To attach to the container:
# docker exec -it $CONTAINER_NAME bash
