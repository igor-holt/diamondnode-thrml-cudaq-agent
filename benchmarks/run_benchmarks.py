#!/usr/bin/env python3
"""Benchmark orchestrator for diamondnode-thrml-cudaq-agent.

Usage:
  python -m benchmarks.run_benchmarks --suite all            # hash+llm+cudaq+thrml
  python -m benchmarks.run_benchmarks --suite cudaq --sizes 4 6 8
  python -m benchmarks.run_benchmarks --suite all --update-baselines
  python -m benchmarks.run_benchmarks --suite llm --check   # regression check, exit 1 on breach

Results go to benchmarks/results/<suite>.json; baselines to baselines/diamondnode-gtx1650.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.storage import BaselineStore, DEFAULT_BASELINE_DIR, summarize, exit_code_for  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SUITES = {
    "hash": {
        "module": "benchmarks.bench_hash",
        "fn": "run_hash_benchmark",
        "args": {},
        "env": "diamondnode-gtx1650-python3.12",
    },
    "llm": {
        "module": "benchmarks.bench_llm",
        "fn": "run_llm_benchmark",
        "args": {"trials": 3},
        "env": "diamondnode-gtx1650-ollama",
    },
    "cudaq": {
        "module": "benchmarks.bench_cudaq",
        "fn": "run_cudaq_benchmark",
        "args": {"shots": 4096},
        "env": "diamondnode-gtx1650-cudaq0.14.2",
    },
    "thrml": {
        "module": "benchmarks.bench_thrml",
        "fn": "run_thrml_benchmark",
        "args": {"samples": 256},
        "env": "diamondnode-gtx1650-thrml-jax",
    },
}

# metric -> pipe-delimited path into result dict ("results|-1|wall_s" = last row)
METRIC_PATHS: dict[str, dict[str, str]] = {
    "hash": {"ops_per_s": "ops_per_s", "p50_ms": "p50_ms"},
    "llm": {"llm_tok_per_s_decode": "results|qwen2:0.5b|tok_per_s_decode_mean",
            "llm_ms_per_token_decode_p50": "results|qwen2:0.5b|ms_per_token_decode_p50"},
    "cudaq": {"cudaq_largest_wall_s": "results|-1|wall_s_mean",
              "cudaq_n10_gap_mean": "results|3|relative_gap_mean"},
    "thrml": {"thrml_largest_samples_per_s": "results|-1|samples_per_s"},
}

TOLERANCES: dict[str, dict[str, dict]] = {
    "hash": {"ops_per_s": {"type": "higher", "limit": 0.50},  # fleet load swings 2x
             "p50_ms": {"type": "lower", "limit": 0.15}},
    "llm": {"llm_tok_per_s_decode": {"type": "higher", "limit": 0.15},
            "llm_ms_per_token_decode_p50": {"type": "lower", "limit": 0.20}},
    "cudaq": {"cudaq_largest_wall_s": {"type": "lower", "limit": 0.25},
              "cudaq_n10_gap_mean": {"type": "lower", "limit": 0.50}},
    "thrml": {"thrml_largest_samples_per_s": {"type": "higher", "limit": 0.25}},
}


def run_suite(name: str, extra_args: dict | None = None) -> dict:
    import importlib
    spec = SUITES[name]
    mod = importlib.import_module(spec["module"])
    args = dict(spec["args"])
    args.update(extra_args or {})
    result = getattr(mod, spec["fn"])(**args)
    result["_meta"] = {
        "suite": name,
        "env": spec["env"],
        "run_at": datetime.now(timezone.utc).isoformat(),
        "args": args,
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="diamondnode benchmark runner")
    p.add_argument("--suite", choices=list(SUITES) + ["all"], default="all")
    p.add_argument("--sizes", type=int, nargs="*", default=None)
    p.add_argument("--spins", type=int, nargs="*", default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--shots", type=int, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--check", action="store_true",
                   help="compare vs baselines, exit nonzero on regression")
    p.add_argument("--update-baselines", action="store_true",
                   help="write results into the baseline file")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_DIR / "diamondnode-gtx1650.json")
    p.add_argument("--env", default="", help="override env tag")
    args = p.parse_args(argv)

    store = BaselineStore(args.baseline)
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    summary_all = {}
    for name in suites:
        extra = {}
        if args.sizes:
            extra["sizes"] = args.sizes
        if args.spins:
            extra["spins"] = args.spins
        if args.trials:
            extra["trials"] = args.trials
        if args.shots:
            extra["shots"] = args.shots
        if args.samples:
            extra["samples"] = args.samples
        print(f"[bench] running suite: {name}")
        result = run_suite(name, extra)
        out = RESULTS_DIR / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[bench] wrote {out}")

        env = args.env or SUITES[name]["env"]
        if args.update_baselines:
            recorded = store.update_from(result, name, METRIC_PATHS[name], env=env)
            print(f"[bench] baselines updated: {recorded} -> {store.path}")
        if args.check:
            rows = store.compare(result, name, METRIC_PATHS[name], TOLERANCES[name])
            summary = summarize(rows)
            summary_all[name] = summary
            print(json.dumps(summary, indent=2))

    if args.check:
        total = {"checked": 0, "regressions": 0}
        for s in summary_all.values():
            total["checked"] += s["checked"]
            total["regressions"] += s["regressions"]
        print(f"[bench] TOTAL: {total['regressions']} regressions / {total['checked']} metrics")
        return 1 if total["regressions"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
