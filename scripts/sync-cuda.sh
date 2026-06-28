#!/bin/bash

uv sync

# Remove the existing cv2.abi3.so file and create a symbolic link to the CUDA-enabled OpenCV shared object file.
rm -f .venv/lib/python3.12/site-packages/cv2/cv2.abi3.so

# Creating a symbolic link to the CUDA-enabled OpenCV shared object file (cv2-cuda-13.2-sm_120.so) in the virtual environment's site-packages directory.
ln -sf $(pwd)/packages/cv2-cuda-13.2-sm_120.so .venv/lib/python3.12/site-packages/cv2/cv2.abi3.so
