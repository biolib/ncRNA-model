#!/bin/bash
docker build --platform linux/amd64  -f Dockerfile -t homology_species .
docker run -it \
    -v /Users/jacoblenzing/Desktop/Thesis/dataset/non_coding_sequences_mmseqs_filtered:/app/non_coding_sequences \
    -v /Users/jacoblenzing/Desktop/Thesis/dataset/homology_app/combined_db:/app/combined_db \
    -v /Users/jacoblenzing/Desktop/Thesis/dataset/homology_app/conserved_rnas:/app/conserved_rnas \
    homology_species /bin/bash