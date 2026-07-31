#!/usr/bin/env python3
"""Thrml Daemon for diamondnode (GTX 1650).

Integrates quantized LLM (Ollama) + THRML Gibbs/thermal sampling (JAX/CUDA)
+ CUDA-Q QUBO routing. Logs successful goal-aligned inference ops to a
JSON ledger and exposes a tiny health endpoint for qmem.

Every backend has a dependency-free fallback so the daemon runs anywhere
(CI, laptop) and upgrades automatically when thrml / cudaq / jax are present.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("thrml_daemon")

warnings.filterwarnings("ignore")

# Optional heavy backends — imported lazily so import never fails.
_jax = None
_jnp = None
_thrml = None
_ising = None
_cudaq = None


def _load_backends():
    global _jax, _jnp, _thrml, _ising, _cudaq
    if _jax is None:
        try:
            import jax
            import jax.numpy as jnp
            _jax, _jnp = jax, jnp
        except ImportError:
            _jax, _jnp = False, False
    if _thrml is None and _jax:
        try:
            from thrml import Block, SamplingSchedule, SpinNode, sample_states
            from thrml.models import IsingEBM, IsingSamplingProgram
            _thrml = (Block, SamplingSchedule, SpinNode, sample_states)
            _ising = (IsingEBM, IsingSamplingProgram)
        except ImportError:
            _thrml, _ising = False, False
    if _cudaq is None:
        try:
            import cudaq
            _cudaq = cudaq
        except ImportError:
            _cudaq = False
    return {
        "jax": bool(_jax),
        "thrml": bool(_thrml),
        "cudaq": bool(_cudaq),
        "jax_devices": [str(d) for d in _jax.devices()] if _jax else [],
    }


def backend_status() -> dict:
    return _load_backends()


OPS_FILE = Path("/dev/shm/thrml_ops.json")
BASELINE_OPS = 0  # new agent cycles; qmem already tracks hash ops


def load_ops(ops_file: Path = OPS_FILE) -> dict:
    if ops_file.exists():
        try:
            return json.loads(ops_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"total_successful_ops": BASELINE_OPS, "last_updated": None, "history": []}


def save_ops(data: dict, ops_file: Path = OPS_FILE) -> None:
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    ops_file.write_text(json.dumps(data, indent=2, default=str))


def record_successful_op(intent: str, tokens: int = 0, energy: float = 0.0,
                         backend: str = "cpu", ops_file: Path = OPS_FILE) -> int:
    data = load_ops(ops_file)
    data["total_successful_ops"] += 1
    data["history"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "intent": intent[:128],
        "tokens": tokens,
        "energy": float(energy),
        "backend": backend,
    })
    data["history"] = data["history"][-1000:]
    save_ops(data, ops_file)
    log.info("Recorded op #%d: %s (tokens=%d energy=%.4f backend=%s)",
             data["total_successful_ops"], intent[:64], tokens, energy, backend)
    return data["total_successful_ops"]


def ising_energy(spins: list[int], biases: list[float], edges, weights: list[float]) -> float:
    """Classical Ising energy: -sum_i b_i s_i - sum_(i,j) w_ij s_i s_j (pure Python)."""
    e = 0.0
    for i, b in enumerate(biases):
        e -= b * spins[i]
    for (i, j), w in zip(edges, weights):
        e -= w * spins[i] * spins[j]
    return e


def thermal_sample_ising(num_spins: int = 8, beta: float = 1.0, samples: int = 64, seed: int = 0) -> dict:
    """THRML block-Gibbs sampling over a cycle graph Ising model.

    Uses THRML when available, else a manual JAX Gibbs loop, else a
    deterministic numpy-free fallback. Always returns the same shape:
    {"samples": [[s0, s1, ...], ...], "energies": [...], "backend": str}
    """
    b = backend_status()
    if b["thrml"]:
        return _thermal_thrml(num_spins, beta, samples, seed)
    if b["jax"]:
        return _thermal_jax(num_spins, beta, samples, seed)
    return _thermal_fallback(num_spins, beta, samples, seed)


def _thermal_thrml(num_spins: int, beta: float, samples: int, seed: int) -> dict:
    jnp = _jnp
    Block, SamplingSchedule, SpinNode, sample_states = _thrml
    IsingEBM, IsingSamplingProgram = _ising
    import jax as _jax_mod
    from thrml import make_empty_block_state
    nodes = [SpinNode() for _ in range(num_spins)]
    edges = [(nodes[i], nodes[(i + 1) % num_spins]) for i in range(num_spins)]
    biases = jnp.array([0.1 * (i % 3) - 0.05 for i in range(num_spins)])
    weights = jnp.array([-1.0 + 0.05 * (i % 2) for i in range(len(edges))])
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    half = max(1, num_spins // 2)
    blocks = [Block(nodes[:half]), Block(nodes[half:])]
    program = IsingSamplingProgram(ebm, blocks, [])
    init_states = make_empty_block_state(blocks, ebm.node_shape_dtypes)
    schedule = SamplingSchedule(max(5, samples // 4), samples, 1)
    t0 = time.perf_counter()
    out = sample_states(_jax_mod.random.PRNGKey(seed), program, schedule,
                        init_states, [], blocks)
    dt = time.perf_counter() - t0
    # block states are bool per block: out[b] shape [n_samples, nodes_in_block]
    n_blocks = len(out)
    spins = []
    for i in range(samples):
        row = []
        for b in range(n_blocks):
            block_i = out[b][i]
            row.extend(int(v) for v in (block_i.flatten().tolist() if hasattr(block_i, "flatten") else [block_i]))
        spins.append([1 if v else -1 for v in row[:num_spins]])
    b = biases.tolist()
    w = weights.tolist()
    idx_edges = [(i, (i + 1) % num_spins) for i in range(num_spins)]
    energies = [ising_energy(s, b, idx_edges, w) for s in spins]
    return {
        "samples": spins,
        "energies": energies,
        "backend": "thrml-jax",
        "wall_s": round(dt, 4),
        "samples_per_s": round(samples / dt, 2) if dt > 0 else 0.0,
        "mean_energy": round(sum(energies) / len(energies), 4) if energies else 0.0,
    }


def _thermal_jax(num_spins: int, beta: float, samples: int, seed: int) -> dict:
    """Manual single-site Gibbs sampler on a cycle graph (JAX), no thrml needed."""
    jnp = _jnp
    rng = _jax.random.PRNGKey(seed)
    biases = jnp.array([0.1 * (i % 3) - 0.05 for i in range(num_spins)])
    weights = jnp.array([-1.0 + 0.05 * (i % 2) for i in range(num_spins)])

    @_jax.jit
    def _gibbs_cycle(x, b, w, r):
        state = x
        for i in range(num_spins):
            j_left, j_right = (i - 1) % num_spins, (i + 1) % num_spins
            field = b[i] + w[j_left] * state[j_left] + w[i] * state[j_right]
            p_up = 1.0 / (1.0 + jnp.exp(-2.0 * beta * field))
            flip = _jax.random.uniform(r, ()) < p_up
            state = state.at[i].set(jnp.where(flip, 1.0, -1.0))
            r, _ = _jax.random.split(r)
        return state

    @_jax.jit
    def _gen(key):
        keys = _jax.random.split(key, samples)
        def _scan_step(carry, key):
            x = carry
            x = _gibbs_cycle(x, biases, weights, key)
            return x, x
        x0 = jnp.ones((num_spins,))
        return _jax.lax.scan(_scan_step, x0, keys)

    t0 = time.perf_counter()
    _, all_x = _gen(rng)
    dt = time.perf_counter() - t0
    spins = jnp.where(all_x > 0, 1, 0).astype(int).tolist()
    b = biases.tolist()
    w = weights.tolist()
    edges = [(i, (i + 1) % num_spins) for i in range(num_spins)]
    energies = [ising_energy(s, b, edges, w) for s in spins]
    return {
        "samples": spins,
        "energies": energies,
        "backend": "jax-gibbs",
        "wall_s": round(dt, 4),
        "samples_per_s": round(samples / dt, 2) if dt > 0 else 0.0,
        "mean_energy": round(sum(energies) / len(energies), 4) if energies else 0.0,
    }


def _thermal_fallback(num_spins: int, beta: float, samples: int, seed: int) -> dict:
    import math
    import random as _rnd
    rng = _rnd.Random(seed)
    edges = [(i, (i + 1) % num_spins) for i in range(num_spins)]
    biases = [0.1 * (i % 3) - 0.05 for i in range(num_spins)]
    weights = [-1.0 + 0.05 * (i % 2) for i in range(len(edges))]
    out = []
    x = [1] * num_spins
    for _ in range(samples):
        for i in range(num_spins):
            j_left, j_right = (i - 1) % num_spins, (i + 1) % num_spins
            field = biases[i] + weights[j_left] * x[j_left] + weights[i] * x[j_right]
            p_up = 1.0 / (1.0 + math.exp(-2.0 * beta * field))
            x[i] = 1 if rng.random() < p_up else -1
        out.append(list(x))
    energies = [ising_energy(s, biases, edges, weights) for s in out]
    return {
        "samples": out,
        "energies": energies,
        "backend": "python-fallback",
        "wall_s": 0.0,
        "samples_per_s": 0.0,
        "mean_energy": round(sum(energies) / len(energies), 4),
    }


def cudaq_qubo_solve(Q_matrix: list[list[float]], shots: int = 2000, seed: int = 7) -> dict:
    """Solve a QUBO min_{x in {0,1}^n} x^T Q x on CUDA-Q (nvidia target when present).

    QUBO x^T Q x maps to Ising energy 1/4 * s^T J s + 1/2 * d^T s + c with
    J[i][j] = Q[i][j] + Q[j][i], d[i] = sum_j (Q[i][j]+Q[j][i]), c = sum Q.
    Uses a depth-1 QAOA-style ansatz (Hadamard + ZZ mixing) seeded by the
    J/d structure; falls back to brute force when cudaq is unavailable.
    """
    n = len(Q_matrix)
    b = backend_status()
    if b["cudaq"]:
        try:
            return _cudaq_qubo_cudaq(Q_matrix, shots)
        except Exception as exc:  # pragma: no cover
            log.warning("cudaq solve failed (%s); falling back to brute force", exc)
    return _qubo_bruteforce(Q_matrix)


_cudaq_bit_order = None  # calibrated once per process: 1 = MSB-first, -1 = LSB-first


def _result_counts(result) -> dict:
    """Extract {bitstring: count} from a CUDA-Q 0.14 SampleResult."""
    items = getattr(result, "items", None)

    def _to_int(v):
        if isinstance(v, dict):
            return sum(int(x) for x in v.values())
        return int(v)

    if items is not None:
        if isinstance(items, dict):
            return {str(k): _to_int(v) for k, v in items.items()}
        try:
            return {str(k): _to_int(v) for k, v in items()}
        except TypeError:
            pass
    try:
        return {str(k): _to_int(v) for k, v in result.get_register_counts().items()}
    except Exception:
        return {}


def _calibrate_bit_order(Q_matrix) -> int:
    """Bit order calibration against brute force at n=4 (CUDA-Q endianness varies)."""
    global _cudaq_bit_order
    n = len(Q_matrix)
    if n != 4:
        return _cudaq_bit_order or 1
    gf = _qubo_bruteforce(Q_matrix)
    ground = gf["ground_energy"]
    best_err, best_order = None, 1
    for order in (1, -1):
        top, _, _ = _cudaq_qubo_sample(Q_matrix, shots=512)
        bits = [int(c) for c in top[::order]]
        err = abs(_qubo_energy(Q_matrix, bits) - ground)
        if best_err is None or err < best_err:
            best_err, best_order = err, order
    _cudaq_bit_order = best_order
    return best_order


def _cudaq_qubo_sample(Q_matrix, shots):
    """Sample the QAOA-style ansatz; returns (top_bitstring, counts, probs).

    Jn is passed as a kernel argument (flat list) — CUDA-Q 0.14 cannot marshal
    closures like 2D lists (NVIDIA/cuda-quantum#4847).
    """
    cudaq = _cudaq
    n = len(Q_matrix)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            q = Q_matrix[i][j]
            if i != j:
                J[i][j] += q
    Jflat = [J[i][j] / 4.0 for i in range(n) for j in range(n)]

    @cudaq.kernel
    def qubo_kernel(n: int, jflat: list[float]):
        q = cudaq.qvector(n)
        for i in range(n):
            h(q[i])
        for i in range(n):
            for j in range(i + 1, n):
                # ZZ(theta) via cx-rz-cx; zero couplings are identity.
                cx(q[i], q[j])
                rz(2.0 * jflat[i * n + j], q[j])
                cx(q[i], q[j])
        for i in range(n):
            rx(0.6, q[i])

    result = cudaq.sample(qubo_kernel, n, Jflat, shots_count=shots)
    counts = _result_counts(result)
    probs = {k: v / shots for k, v in counts.items()}
    top = result.most_probable()
    if not top and counts:
        top = max(counts, key=counts.get)
    return top, counts, probs


def _cudaq_qubo_cudaq(Q_matrix, shots) -> dict:
    cudaq = _cudaq
    n = len(Q_matrix)
    t0 = time.perf_counter()
    try:
        cudaq.set_target("nvidia")
    except Exception:
        pass
    targets = cudaq.get_targets()
    has_nvidia = targets is not None and any(
        getattr(t, "name", "") == "nvidia" for t in targets)
    if not has_nvidia:
        try:
            cudaq.set_target("qpp-cpu")
        except Exception:
            pass

    order = _calibrate_bit_order(Q_matrix)
    top, counts, probs = _cudaq_qubo_sample(Q_matrix, shots)
    dt = time.perf_counter() - t0
    top_bits = [int(c) for c in top[::order]]
    energy_qubo = _qubo_energy(Q_matrix, top_bits)
    return {
        "backend": "cudaq-" + cudaq.get_target().name,
        "n": n,
        "shots": shots,
        "wall_s": round(dt, 4),
        "top_sample": top_bits,
        "top_qubo_energy": round(energy_qubo, 6),
        "top_count": max(counts.values()) if counts else 0,
        "top_probability": round(max(probs.values()), 6) if probs else 0.0,
        "samples_per_s": round(shots / dt, 1) if dt > 0 else 0.0,
        "ground_energy": None,
        "optimal": False,
        "bit_order": order,
    }


def _qubo_energy(Q, x) -> float:
    n = len(Q)
    return sum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n))


def _qubo_bruteforce(Q_matrix) -> dict:
    n = len(Q_matrix)
    best_e, best_x = None, None
    for mask in range(1 << n):
        x = [(mask >> i) & 1 for i in range(n)]
        e = _qubo_energy(Q_matrix, x)
        if best_e is None or e < best_e:
            best_e, best_x = e, x
    return {
        "backend": "bruteforce",
        "n": n,
        "shots": 1 << n,
        "wall_s": 0.0,
        "top_sample": best_x,
        "top_qubo_energy": round(best_e, 6),
        "top_count": 1,
        "top_probability": 1.0,
        "samples_per_s": 0.0,
        "ground_energy": round(best_e, 6),
        "optimal": True,
    }


def ollama_generate(prompt: str, model: str = "qwen2:0.5b",
                    base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0) -> dict:
    """Generate tokens via the local Ollama API (no auth needed on node)."""
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt,
                          "stream": False, "options": {"num_predict": 64}}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    dt = time.perf_counter() - t0
    n_tokens = int(data.get("eval_count", 0))
    return {
        "model": model,
        "tokens": n_tokens,
        "wall_s": round(dt, 3),
        "tok_per_s": round(n_tokens / dt, 2) if dt > 0 else 0.0,
        "text": (data.get("response", "") or "")[:120],
    }


def agent_cycle(goal_intent: str, use_llm: bool = False, model: str = "qwen2:0.5b",
                num_spins: int = 8, beta: float = 1.0, ops_file: Path = OPS_FILE) -> dict:
    """One successful goal-driven inference cycle.

    1. Optional LLM forward (Ollama) — tokens counted.
    2. THRML / JAX thermal sample for decision routing.
    3. CUDA-Q QUBO solve of a tiny routing subproblem.
    4. Record aligned op in the ledger.
    """
    b = backend_status()
    llm = ollama_generate(goal_intent, model=model) if use_llm else {"tokens": 0}
    thermal = thermal_sample_ising(num_spins=num_spins, beta=beta)
    n = min(num_spins, 8)
    import random as _rnd
    r = _rnd.Random(42)
    Q = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Q[i][i] = r.uniform(-0.5, 0.5)
        for j in range(i + 1, n):
            v = r.uniform(-0.2, 0.2)
            Q[i][j] = Q[j][i] = v
    qubo = cudaq_qubo_solve(Q)
    energy = thermal["mean_energy"]
    op_num = record_successful_op(
        goal_intent, tokens=int(llm.get("tokens", 0)), energy=energy,
        backend=thermal["backend"], ops_file=ops_file)
    return {
        "status": "success",
        "op_number": op_num,
        "backends": b,
        "llm": llm,
        "thermal": thermal,
        "qubo": qubo,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="THRML daemon (diamondnode GTX 1650)")
    p.add_argument("--ops-log", type=Path, default=OPS_FILE, help="ops ledger JSON path")
    p.add_argument("--intent", default="hybridization: deploy local agent network with THRML sampling")
    p.add_argument("--use-llm", action="store_true", help="generate tokens via Ollama")
    p.add_argument("--model", default="qwen2:0.5b")
    p.add_argument("--num-spins", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--status", action="store_true", help="print backend status and exit")
    args = p.parse_args(argv)
    if args.status:
        print(json.dumps(backend_status(), indent=2))
        return 0
    log.info("Thrml Daemon starting on diamondnode substrate")
    result = agent_cycle(args.intent, use_llm=args.use_llm, model=args.model,
                         num_spins=args.num_spins, beta=args.beta, ops_file=args.ops_log)
    print(json.dumps(result, indent=2, default=str))
    print("Ops file:", args.ops_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
