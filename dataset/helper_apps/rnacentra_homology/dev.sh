#!/bin/bash
docker build --platform linux/amd64  -f Dockerfile -t rnacentral_homology .
docker run -it \
    rnacentral_homology /bin/bash