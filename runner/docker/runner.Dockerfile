FROM ubuntu:jammy-20240227

RUN apt-get update && apt-get install -y  \
    curl \
    tar \
    apt-transport-https \
    libtiff5 \
    libicu66 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m nonroot
