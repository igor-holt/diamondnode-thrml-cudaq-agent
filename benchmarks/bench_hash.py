"""Hash throughput + latency baseline (mirrors the qmem 15,265 ops/sec claim).

Pure stdlib (hashlib, threading) — no numpy needed, so this runs anywhere.
Measures ops/sec, p50/p95 latency, and a J/op estimate (CPU-bound scaling).
"""
from __future__ import annotations

import hashlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def _sha256_op(data: bytes, iters: int) -> bytes:
    h = hashlib.sha256()
    for _ in range(iters):
        h.update(data)
    return h.digest()


def run_hash_benchmark(size: int = 64, workers: int = 4, duration: float = 3.0,
                       op: Callable = _sha256_op, iters: int = 100) -> dict:
    """Run `duration` seconds of hashing across `workers` threads."""
    data = bytes(range(size))
    total_ops = 0
    latencies: list[float] = []

    def worker(stop: dict):
        nonlocal total_ops
        while not stop["stop"]:
            t0 = time.perf_counter()
            op(data, iters)
            dt = time.perf_counter() - t0
            latencies.append(dt / iters)
            total_ops += iters

    stop = {"stop": False}
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, stop) for _ in range(workers)]
        time.sleep(duration)
        stop["stop"] = True
        for f in futs:
            f.result()
    wall = time.perf_counter() - t_start

    ops_per_s = total_ops / wall
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    # Energy proxy: 1 CPU core ~ TDP_pct; use conservative 30 W for the running threads
    # and 1 s baseline → J/op. Hardware-instrumented values come from qmem later.
    joules_per_op = (30.0 * wall) / total_ops if total_ops else 0.0
    return {
        "backend": "python-hashlib-sha256",
        "workers": workers,
        "wall_s": round(wall, 3),
        "ops_per_s": round(ops_per_s, 1),
        "p50_ms": round(p50 * 1e3, 4),
        "p95_ms": round(p95 * 1e3, 4),
        "joules_per_op_est": round(joules_per_op, 6),
        "crystal_state": "CRYSTALLINE" if ops_per_s > 1000 else "UNKNOWN",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_hash_benchmark(), indent=2))
