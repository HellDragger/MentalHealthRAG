"""MHRAG: safety-aware, citation-grounded RAG for mental-health psychoeducation."""

import os

# This project uses PyTorch only. On images that also ship TensorFlow and JAX (Kaggle, Colab), a library that touches
# them can make JAX reserve 75% of the GPU for itself, which then runs PyTorch out of memory (seen on Kaggle T4s:
# ~11 GB held outside PyTorch). Keep both off the GPU unless the caller has set these explicitly.
for _k, _v in {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "USE_TF": "0",  # transformers / datasets: do not import TensorFlow
    "USE_JAX": "0",
    "USE_FLAX": "0",
}.items():
    os.environ.setdefault(_k, _v)
