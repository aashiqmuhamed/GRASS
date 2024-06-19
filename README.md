# GRASS: Compute Efficient Low-Memory LLM Training with Structured Sparse Gradients

This repository contains the code to reproduce [GRASS: Compute Efficient Low-Memory LLM Training with Structured Sparse Gradients]().

GRASS (GRAdient Structured Sparsification) introduces sparse projections to transform gradients into structured sparse updates, significantly reducing memory usage for optimizer states and minimizing gradient memory footprint, computation, and communication costs. This approach enables half-precision pretraining of a 13B parameter LLaMA model on a single 40GB A100 GPU and achieves up to a $2\times$ throughput improvement on an 8-GPU system, while maintaining comparable performance to full-rank training and existing projection-based methods.
