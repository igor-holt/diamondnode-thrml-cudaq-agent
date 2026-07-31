"""CUDA-Q QUBO benchmark: wall time + solution quality vs brute-force ground state.

Solves fixed-seed random QUBOs at increasing sizes on the `nvidia` target
(GTX 1650) via the daemon's QAOA-style kernel; falls back to `qpp-cpu`.
Solution quality = relative gap to the brute-force optimum (computed for n <= 10).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thrml_daemon import cudaq_qubo_solve, _qubo_energy, _qubo_bruteforce  # noqa: E402

DEFAULT_SIZES = [4, 6, 8, 10, 12, 14]
DEFAULT_SHOTS = 4096
DEFAULT_REPEATS = 3  # cudaq.sample is unseeded (no set_seed in 0.14) — average repeats


def make_qubo(n: int, seed: int = 7, density: float = 0.6) -> list[list[float]]:
    """Random symmetric QUBO (Q_ij = Q_ji), zero diagonal excluded for sparsity."""
    r = random.Random(seed)
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if r.random() < density:
                v = r.uniform(-1.0, 1.0)
                Q[i][j] = Q[j][i] = v
    return Q


def run_cudaq_benchmark(sizes: list[int] | None = None, shots: int = DEFAULT_SHOTS,
                        seed: int = 7, repeats: int = DEFAULT_REPEATS) -> dict:
    sizes = sizes or DEFAULT_SIZES
    rows = []
    for n in sizes:
        Q = make_qubo(n, seed=seed)
        repeat_rows = []
        for rep in range(repeats):
            t0 = time.perf_counter()
            res = cudaq_qubo_solve(Q, shots=shots, seed=seed)
            res["prep_s"] = round(time.perf_counter() - t0, 4)
            res["repeat"] = rep
            repeat_rows.append(res)
        # collapse repeats: mean wall + best energy/gap (any-repeat optimum)
        best = min(repeat_rows, key=lambda r: r["top_qubo_energy"])
        row = {
            "backend": res["backend"],
            "n": n,
            "shots": shots,
            "repeats": repeats,
            "wall_s_mean": round(sum(r["wall_s"] for r in repeat_rows) / repeats, 4),
            "wall_s_best": min(r["wall_s"] for r in repeat_rows),
            "top_qubo_energy_best": best["top_qubo_energy"],
            "top_sample_best": best["top_sample"],
            "samples_per_s_mean": round(
                sum(r["samples_per_s"] for r in repeat_rows) / repeats, 1),
            "top_probability_best": best["top_probability"],
            "bit_order": best.get("bit_order"),
            "repeats_detail": [{k: v for k, v in r.items() if k not in ("top_sample",)}
                               for r in repeat_rows],
        }
        if n <= 10:
            gf = _qubo_bruteforce(Q)
            ground = gf["ground_energy"]
            gap = abs(best["top_qubo_energy"] - ground) / max(abs(ground), 1e-9)
            row["ground_energy"] = ground
            row["relative_gap_best"] = round(gap, 5)
            row["optimal"] = abs(gap) < 1e-6
            gaps = [abs(r["top_qubo_energy"] - ground) / max(abs(ground), 1e-9)
                    for r in repeat_rows]
            row["relative_gap_mean"] = round(sum(gaps) / repeats, 5)
        rows.append(row)
    return {"backend": rows[0]["backend"] if rows else "none", "shots": shots,
            "repeats": repeats, "results": rows}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    p.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = p.parse_args()
    print(json.dumps(run_cudaq_benchmark(args.sizes, args.shots, repeats=args.repeats), indent=2))
