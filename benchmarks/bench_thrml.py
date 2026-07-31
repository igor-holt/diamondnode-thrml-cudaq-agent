"""THRML block-Gibbs sampling benchmark (JAX/CUDA on GTX 1650, CPU fallback).

Measures samples/sec and mean-energy convergence for cycle-graph Ising EBMs
at increasing spin counts, using the daemon's real thrml pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thrml_daemon import thermal_sample_ising, backend_status  # noqa: E402

DEFAULT_SPINS = [4, 8, 16, 32, 64]
DEFAULT_SAMPLES = 256
DEFAULT_BETA = 1.0


def run_thrml_benchmark(spins: list[int] | None = None, samples: int = DEFAULT_SAMPLES,
                        beta: float = DEFAULT_BETA, seed: int = 0) -> dict:
    spins = spins or DEFAULT_SPINS
    rows = []
    for n in spins:
        res = thermal_sample_ising(num_spins=n, beta=beta, samples=samples, seed=seed)
        rows.append({
            "n_spins": n,
            "backend": res["backend"],
            "samples": len(res["samples"]),
            "wall_s": res["wall_s"],
            "samples_per_s": res["samples_per_s"],
            "mean_energy": res["mean_energy"],
            "energy_std": round(
                (sum((e - res["mean_energy"]) ** 2 for e in res["energies"]) / len(res["energies"])) ** 0.5, 4
            ) if res["energies"] else 0.0,
        })
    return {"backend": rows[0]["backend"] if rows else "none", "devices": backend_status()["jax_devices"], "results": rows}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spins", type=int, nargs="*", default=DEFAULT_SPINS)
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = p.parse_args()
    print(json.dumps(run_thrml_benchmark(args.spins, args.samples), indent=2))
