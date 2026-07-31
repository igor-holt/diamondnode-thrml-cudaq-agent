#!/usr/bin/env python3
"""CPU-safe tests for benchmarks (baseline store, hash bench, qubo builder)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.storage import BaselineStore, resolve_path, summarize  # noqa: E402
from benchmarks.bench_hash import run_hash_benchmark  # noqa: E402
from benchmarks.bench_cudaq import make_qubo  # noqa: E402
from benchmarks import run_benchmarks  # noqa: E402


def test_store_set_and_get(tmp_path):
    s = BaselineStore(tmp_path / "b.json")
    s.set_metric("hash", "ops_per_s", 15265.0, unit="ops/s", env="node")
    s2 = BaselineStore(tmp_path / "b.json")
    assert s2.metric("hash", "ops_per_s")["value"] == 15265.0


def test_store_update_from_nested(tmp_path):
    s = BaselineStore(tmp_path / "b.json")
    result = {"results": [{"a": 1.0}, {"a": 2.0}]}
    recorded = s.update_from(result, "cudaq", {"last_a": "results|-1|a"}, env="node")
    assert recorded == {"last_a": 2.0}
    assert s.metric("cudaq", "last_a")["value"] == 2.0


def test_resolve_path(tmp_path):
    assert resolve_path({"a": {"b": [5, 6]}}, "a|b|-1") == 6
    assert resolve_path({"a": {"b": [5, 6]}}, "a|b|0") == 5
    assert resolve_path({"a": {}}, "a|b|c") is None


def test_compare_regression_detection(tmp_path):
    s = BaselineStore(tmp_path / "b.json")
    s.set_metric("hash", "ops_per_s", 1000.0)
    rows = s.compare("hash", {"ops_per_s": 800.0}, {"ops_per_s": "ops_per_s"},
                     {"ops_per_s": {"type": "higher", "limit": 0.10}})
    assert rows[0]["status"] == "regression"
    rows_ok = s.compare("hash", {"ops_per_s": 1100.0}, {"ops_per_s": "ops_per_s"},
                        {"ops_per_s": {"type": "higher", "limit": 0.10}})
    assert rows_ok[0]["status"] == "ok"


def test_compare_lower_is_better(tmp_path):
    s = BaselineStore(tmp_path / "b.json")
    s.set_metric("llm", "ms_per_token_p50", 20.0)
    rows = s.compare("llm", {"ms_per_token_p50": 30.0}, {"ms_per_token_p50": "ms_per_token_p50"},
                     {"ms_per_token_p50": {"type": "lower", "limit": 0.15}})
    assert rows[0]["status"] == "regression"


def test_compare_no_baseline(tmp_path):
    s = BaselineStore(tmp_path / "b.json")
    rows = s.compare("llm", {"tok": 5.0}, {"tok": "tok"}, {})
    assert rows[0]["status"] == "no-baseline"


def test_summarize_counts():
    s = summarize([{"status": "ok"}, {"status": "regression"}, {"status": "no-baseline"}])
    assert s == {"checked": 3, "regressions": 1, "no_baseline": 1, "ok": 1,
                 "details": [{"status": "ok"}, {"status": "regression"}, {"status": "no-baseline"}]}


def test_hash_benchmark_runs(tmp_path):
    res = run_hash_benchmark(size=32, workers=2, duration=0.5)
    assert res["ops_per_s"] > 0
    assert res["p50_ms"] > 0
    assert res["crystal_state"] == "CRYSTALLINE"


def test_hash_benchmark_deterministic_metrics():
    a = run_hash_benchmark(size=32, workers=1, duration=0.5)
    b = run_hash_benchmark(size=32, workers=1, duration=0.5)
    assert a["ops_per_s"] == pytest.approx(b["ops_per_s"], rel=0.5)  # loose: machine noise


def test_make_qubo_symmetric_and_seeded():
    Q1 = make_qubo(6, seed=7)
    Q2 = make_qubo(6, seed=7)
    Q3 = make_qubo(6, seed=8)
    assert Q1 == Q2
    assert Q1 != Q3
    for i in range(6):
        for j in range(6):
            assert Q1[i][j] == Q1[j][i]


def test_runner_metrics_paths_valid():
    for suite, paths in run_benchmarks.METRIC_PATHS.items():
        for metric, path in paths.items():
            assert isinstance(path, str) and path


def test_runner_suites_importable():
    import importlib
    for name in run_benchmarks.SUITES:
        mod = importlib.import_module(run_benchmarks.SUITES[name]["module"])
        assert callable(getattr(mod, run_benchmarks.SUITES[name]["fn"]))
