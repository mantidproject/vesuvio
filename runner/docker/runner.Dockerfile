FROM ubuntu:focal-20231128

RUN apt-get update && apt-get install -y \
    curl \
    tar \
    libicu72 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m nonroot