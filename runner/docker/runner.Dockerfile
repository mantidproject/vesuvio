FROM ubuntu:focal-20231128

RUN apt-get update && apt-get install -y  \
    curl \
    tar \
    apt-transport-https \
    libicu66 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m nonroot