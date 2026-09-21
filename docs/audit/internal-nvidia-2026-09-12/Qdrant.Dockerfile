FROM qdrant/qdrant@sha256:6c0652f8d6925b22f2f6f0e0a5365a6c9dbc8768bd6e70ccc1cdc14847e452a0
USER root
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
