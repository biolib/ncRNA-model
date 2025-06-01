#!/bin/bash
docker build -f Dockerfile_noCUDA -t baseline-train-and-evaluate_no_cuda .
docker run -it \
    -v $(pwd)/main.py:/home/biolib/main.py \
    -v $(pwd)/finetune_data:/home/biolib/finetune_data \
    baseline-train-and-evaluate_no_cuda  /bin/bash