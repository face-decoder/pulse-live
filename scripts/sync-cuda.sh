#!/bin/bash

# Sync virtualenv dependencies via uv
uv sync

# Set environment variables for OpenCV CUDA shared libraries
export LD_LIBRARY_PATH="/home/inadio/dev-build/opencv/install/lib64:/home/inadio/dev-build/cuda-env/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

echo "OpenCV CUDA environment synced successfully."

