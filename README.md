# diamondnode-thrml-cudaq-agent

Thrml Daemon + CUDA-Q QUBO hybrid agent stack for Diamondnode (GTX 1650 @ qmem.genesisconductor.io).

## Core
- Primary local model: Qwen/Qwen2.5-1.5B-Instruct GGUF Q4_K_M (~1.1 GB, fits 4 GB VRAM)
- Thermal/Gibbs sampling via Extropic THRML (JAX + CUDA)
- QUBO routing / Ising Hamiltonians via CUDA-Q (nvidia target simulation) + THRML IsingEBM
- Goal-driven inference ops counter (baseline 0 new agent cycles; existing qmem hash ops already at 15k+/s)
- Distillation path from Inkling (cloud) → local LoRA/Unsloth
- Integrates with existing qmem Bubble Gateway / S-ToT / seismic log at https://qmem.genesisconductor.io

## Live Diamondnode
- Endpoint: https://qmem.genesisconductor.io
- GPU: GTX 1650 4 GB
- Verified: 15,265 ops/sec hash, 1.1 ms p50, 0.042 J/op, CRYSTALLINE
- Tunnel: Cloudflare

## Quick Start (on diamondnode)
```bash
# Assume CUDA toolkit, JAX[cuda], llama.cpp CUDA, thrml, cuda-quantum installed
pip install thrml jax[cuda12]  # or matching
# Download GGUF
# huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
python thrml_daemon.py --model path/to/gguf --ops-log /dev/shm/thrml_ops.json
```

## Milestone: Inference Ops
Counter starts at documented baseline from qmem. New successful goal-aligned agent cycles logged here. Target: thousands/day via parallel local agents + cloud supervision (Inkling/Grok/Claude).

## Oahu H-3 / Tunnel-Through
Direct penetration of VRAM mountain via quantized small model + superior sampling + hybrid quantum-thermo routing. Advances Hybridization & Consciousness + Intrinsic Pursuit (EBM/QUBO) + Financial Infrastructure (QUBO economics streams).

evt- protocol + trace-consent + maru guards enforced.
ORCID: 0009-0008-8389-1297
