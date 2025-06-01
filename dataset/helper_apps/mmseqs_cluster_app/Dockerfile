FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    tar \
    gzip \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz && \
    tar xvfz mmseqs-linux-sse41.tar.gz && \
    cp mmseqs/bin/mmseqs /usr/local/bin/ && \
    rm -rf mmseqs mmseqs-linux-sse41.tar.gz

RUN pip install biopython
RUN mkdir tmp_search

COPY launch.py .
COPY mmseqs_cluster.sh .
RUN chmod +x mmseqs_cluster.sh

ENTRYPOINT []