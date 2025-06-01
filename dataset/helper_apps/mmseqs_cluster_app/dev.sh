#!/bin/bash
docker build --platform linux/amd64  -f Dockerfile -t cluster_species .
docker run -it \
    cluster_species /bin/bash