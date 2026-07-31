#!/usr/bin/env python3
"""CPU-safe tests for thrml_daemon (no GPU required — fallback paths)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import thrml_daemon as td  # noqa: E402


@pytest.fixture()
def ops_file(tmp_path):
    return tmp_path / "ops.json"


def test_load_ops_empty(tmp_path):
    data = td.load_ops(tmp_path / "nope.json")
    assert data["total_successful_ops"] == 0
    assert data["history"] == []


def test_record_successful_op_roundtrip(ops_file):
    n = td.record_successful_op("test intent", tokens=12, energy=-3.5, ops_file=ops_file)
    assert n == 1
    data = td.load_ops(ops_file)
    assert data["total_successful_ops"] == 1
    assert data["history"][0]["tokens"] == 12
    assert data["history"][0]["energy"] == -3.5
    assert data["history"][0]["intent"] == "test intent"
    assert data["last_updated"] is not None


def test_record_keeps_last_1000(ops_file):
    for i in range(1010):
        td.record_successful_op(f"op {i}", ops_file=ops_file)
    data = td.load_ops(ops_file)
    assert data["total_successful_ops"] == 1010
    assert len(data["history"]) == 1000
    assert data["history"][0]["intent"] == "op 10"


def test_ising_energy_hand_calculated():
    spins = [1, -1, 1]
    biases = [0.5, -0.5, 0.5]
    edges = [(0, 1), (1, 2), (0, 2)]
    weights = [2.0, -1.0, 0.5]
    # -0.5*1 -(-0.5)(-1) -0.5*1 - 2*(1*-1) - (-1)(-1*1) - 0.5*(1*1)
    expected = -(0.5) - (-0.5 * -1) - (0.5) - (2.0 * -1) - (-1.0 * -1) - (0.5 * 1)
    assert td.ising_energy(spins, biases, edges, weights) == pytest.approx(expected)


def test_thermal_fallback_deterministic_and_valid():
    a = td._thermal_fallback(6, 1.0, 32, seed=3)
    b = td._thermal_fallback(6, 1.0, 32, seed=3)
    assert a == b
    assert len(a["samples"]) == 32
    assert all(len(s) == 6 for s in a["samples"])
    assert all(v in (-1, 1) for s in a["samples"] for v in s)
    assert len(a["energies"]) == 32
    assert a["backend"] == "python-fallback"
    assert a["mean_energy"] < 0.0  # ferromagnetic-ish cycle at beta=1


def test_thermal_high_beta_lower_energy():
    lo = td._thermal_fallback(6, 5.0, 64, seed=3)["mean_energy"]
    hi = td._thermal_fallback(6, 0.05, 64, seed=3)["mean_energy"]
    assert lo < hi  # colder = lower mean energy


def test_qubo_bruteforce_known_optimum():
    # Q = [[1,-1],[-1,1]] with x in {0,1}^2:
    # x=(0,0) -> 0 ; x=(1,1) -> 0 ; x=(1,0) -> 1+... wait: Q[0][0]+Q[1][1]... compute
    Q = [[1.0, -1.0], [-1.0, 1.0]]
    res = td._qubo_bruteforce(Q)
    assert res["optimal"] is True
    # E(0,1)=1, E(1,0)=1, E(1,1)=1-1-1+1=0, E(0,0)=0 -> min 0
    assert res["top_qubo_energy"] == 0.0
    assert res["top_sample"] in ([0, 0], [1, 1])


def test_cudaq_qubo_falls_back_without_cudaq(monkeypatch):
    monkeypatch.setattr(td, "_load_backends", lambda: {"cudaq": False})
    Q = [[1.0, -1.0], [-1.0, 1.0]]
    res = td.cudaq_qubo_solve(Q)
    assert res["backend"] == "bruteforce"
    assert res["optimal"] is True


def test_thermal_sample_shape_any_backend():
    res = td.thermal_sample_ising(num_spins=4, samples=16, seed=1)
    assert len(res["samples"]) == 16
    assert len(res["energies"]) == 16
    assert res["backend"] in ("thrml-jax", "jax-gibbs", "python-fallback")


def test_agent_cycle_records_op(ops_file):
    res = td.agent_cycle("unit test cycle", ops_file=ops_file)
    assert res["status"] == "success"
    assert res["op_number"] == 1
    assert "thermal" in res and "qubo" in res
    assert isinstance(res["backends"], dict)


def test_backend_status_keys():
    s = td.backend_status()
    assert set(s) == {"jax", "thrml", "cudaq", "jax_devices"}


def test_main_status_flag(capsys):
    rc = td.main(["--status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "jax" in out
