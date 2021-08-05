# tensorflow-Sobel-edge-detection
Sobel edge detection via tensorflow

## conda environment

without CUDA:

    conda create -n sobel python=3.9 tensorflow tensorflow-probability

with CUDA

    conda create -n sobel-gpu python=3.9 tensorflow-gpu tensorflow-probability cudatoolkit cudnn

MAKE SURE THE ENVIRONMENT IS ACTIVATED BEFORE RUNNING THE CODE !

## usage

    python sobel.py -I path/to/image -p [float] (percentage of points to remove)
