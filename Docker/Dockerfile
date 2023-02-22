FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  && pip install -q \
  dalle-mini \
  git+https://github.com/kkowalak/dalle-mini.git \
  git+https://github.com/patil-suraj/vqgan-jax.git \
  jax==0.3.25

WORKDIR /workspace
