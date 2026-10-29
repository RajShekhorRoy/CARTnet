FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Linux dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        dssp \
    && rm -rf /var/lib/apt/lists/*

# DSSP 4.x needs the PDB Chemical Component Dictionary
RUN mkdir -p /var/cache/libcifpp && \
    curl -L \
        -o /var/cache/libcifpp/components.cif \
        https://files.wwpdb.org/pub/pdb/data/monomers/components.cif

RUN pip install \
    esm==3.2.1.post1 \
    httpx==0.28.1 \
    pytorch-lightning==2.6.1

# Copy CARTnet
COPY . /app

# Sanity check DSSP exists
RUN which mkdssp && mkdssp --version

ENTRYPOINT ["python", "main.py"]