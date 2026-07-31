# diamondnode-thrml-cudaq-agent

Thrml Daemon + CUDA-Q QUBO hybrid agent stack for Diamondnode (GTX 1650 @ qmem.genesisconductor.io) — with **real benchmark suites, regression checks, and stored baselines**.

## Stack

| Layer | Tech | Status on node |
|-------|------|----------------|
| Local LLM | Ollama (`qwen2:0.5b`, `llama3.2:3b`, …) | live |
| Thermal / Gibbs sampling | Extropic THRML (`IsingEBM` + block Gibbs on JAX/CUDA) | `thrml-jax` on `cuda:0` |
| QUBO routing | CUDA-Q 0.14.2, `nvidia` target (cuStateVec on GTX 1650) | `cudaq-nvidia` |
| Fallbacks | pure-Python/JAX-CPU/brute-force when backends absent | CI-safe |

`thrml_daemon.py` runs anywhere: it lazily loads `thrml` / `cudaq` / `jax` and degrades to
CPU or brute-force fallbacks so tests and CI never need a GPU.

## Daemon

```bash
python thrml_daemon.py --status                    # backend availability
python thrml_daemon.py --ops-log /dev/shm/thrml_ops.json --intent "hybridization: ..."
python thrml_daemon.py --use-llm --model qwen2:0.5b --num-spins 16 --beta 1.0
```

A cycle = optional Ollama tokens → THRML Ising sample (decision routing) → CUDA-Q QUBO
solve (routing subproblem) → op recorded in the JSON ledger with intent, tokens, energy,
backend. Ledger defaults to `/dev/shm/thrml_ops.json`, keeps the last 1000 ops.

## Benchmarks

```bash
python -m benchmarks.run_benchmarks --suite all                      # hash + llm + cudaq + thrml
python -m benchmarks.run_benchmarks --suite cudaq --sizes 4 6 8      # subset
python -m benchmarks.run_benchmarks --suite all --check              # regression gate, exit 1 on breach
python -m benchmarks.run_benchmarks --suite all --update-baselines   # record new baselines
```

Results → `benchmarks/results/<suite>.json`; baselines → `baselines/diamondnode-gtx1650.json`.

Suites:

- **hash** — SHA-256 throughput/latency (pure stdlib), energy proxy, CRYSTALLINE gate
- **llm** — Ollama generate: tok/s, ms/token p50, J/token estimate, per-model resilience
- **cudaq** — fixed-seed random QUBOs, n = 4…14, 4096 shots, QAOA-style kernel on `nvidia`
  target; records wall time, top energy, relative gap vs brute-force ground (n ≤ 10), bit-order calibration
- **thrml** — THRML block-Gibbs on cycle-graph Ising EBM, n = 4…64 spins, 256 samples:
  samples/s, mean energy, energy std, device

## Measured baselines (diamondnode, 2026-07-31)

Machine is a shared fleet host (load 60+, grok llama-server resident on the GPU at
2.8/4 GB VRAM) — numbers below are under live fleet load. CUDA-Q is unseeded in
0.14, so the suite averages 3 repeats per size; LLM decode rates come from
ollama's `eval_duration` (ns fields on this build), separate from wall time.

| Suite | Metric | Value |
|-------|--------|-------|
| hash | ops/s (4 threads, SHA-256) | **1.25–2.35M ops/s** (load-dependent; gate tolerance 50%) |
| hash | p50 latency | **0.0002 ms** |
| llm (qwen2:0.5b) | decode | **141.6 tok/s** (7.1 ms/token p50), 0.36 s load |
| llm (qwen2:0.5b) | end-to-end wall | 74.5 tok/s (includes load + prompt eval) |
| llm (llama3.2:3b) | decode | **0.53 tok/s** — VRAM-starved (CPU spill; grok holds 2.8/4 GB) |
| cudaq (n=14, 4096 shots × 3) | solve wall (warm JIT) | **0.118 s** mean, 34,944 shots/s |
| cudaq (n=4…10) | quality vs brute force | best-3-run gap 0.48–1.41 (mean gap 0.69–1.63); depth-1 fixed-angle QAOA heuristic |
| thrml (n=64, 256 samples) | sampling | **9.4 samples/s** on `cuda:0` (24–38 samples/s for n ≤ 32) |

Full runs: `benchmarks/results/*.json` (re-generable on the node).

### Verified accuracy notes (2026-07-31 audit)
- THRML mean energy / std recompute exactly from stored samples (±1 spins only).
- Daemon backend head-to-head on the same 8-spin cycle Ising model, 256 samples,
  seed 0 (mean energy, lower = better): python fallback −6.37 @ β=1 (deepest —
  its per-site sweep mixes harder), thrml-jax −4.32 (GPU path; add warmup for
  deeper convergence), manual jax-gibbs +0.29 (equilibrates slowly from the
  all-ones start — fallback only, use the thrml path in production).
- CUDA-Q stored energies/gaps recompute to displayed precision; bit-order
  calibration (distribution-based, cached per process) was stable across runs.
- Brute-force scaling measured on-node: n=14 → 382 ms, n=16 → 2.0 s, n=18 →
  10.8 s, n=20 → 52 s ⇒ crossover vs warm-cache CUDA-Q (~0.12 s) at n ≈ 13–14.
- First-call CUDA-Q includes JIT compile (tens of seconds); `wall_s` after
  warm-up is what the baseline tracks.

Known limits (honest):
- `llama3.2:3b` decodes at 0.53 tok/s only because grok's `llama-server`
  occupies the VRAM — with a free GPU slot it should reach tens of tok/s.
  Track `qwen2:0.5b` (GPU-resident, 141 tok/s) by default.
- Depth-1 QAOA with fixed angles is a heuristic: relative gap vs ground truth
  (0.5–1.6 on these instances) is the quality baseline to beat via angle
  optimization, more layers, or THRML-guided sampling.
- `thrml` n=64 mean energy does not converge from the empty init state in 256
  samples at β=1 — increase samples or warmup for production routing.

## Tests & CI

```bash
pip install -r requirements-dev.txt
pytest            # 24 CPU-safe tests, no GPU required
```

`.github/workflows/ci.yml` runs pytest on Python 3.10/3.12 and smoke-tests the daemon +
hash suite in CI.

## Evolution (via qmem / local CSID)

```bash
python3 ~/gc-workers/stream-virtual-experience/scripts/thrml-csid-daemon.py  # :5192
curl -sS http://127.0.0.1:5192/api/evolve?space=1RKZzzEXjXmKB\&steps=3 | jq '{goal_status,whiteRabbit}'
```

evt- protocol + trace-consent + maru guards enforced. ORCID: 0009-0008-8389-1297
