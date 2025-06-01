#!/bin/bash
docker build -f Dockerfile -t baseline-train-and-evaluate_cuda .
docker run -it --gpus all \
    -v $(pwd)/finetune-data:/home/biolib/finetune_data \
    -v $(pwd)/configs:/home/biolib/configs \
    -v $(pwd)/main.py:/home/biolib/main.py \
    -v $(pwd)/10000_rnacentral.fasta:/home/biolib/10000_rnacentral.fasta \
    -v $(pwd)/finetune_mlm_transformer.py:/home/biolib/finetune_mlm_transformer.py \
    -v $(pwd)/finetune_mlm_1dcnn.py:/home/biolib/finetune_mlm_1dcnn.py \
    -v $(pwd)/pretrain_mlm_transformer.py:/home/biolib/pretrain_mlm_transformer.py \
    -v $(pwd)/pretrain_mlm_1dcnn.py:/home/biolib/pretrain_mlm_1dcnn.py \
    -v $(pwd)/api_key:/biolib/secrets/WANDB_API_KEY \
    baseline-train-and-evaluate_cuda /bin/bash