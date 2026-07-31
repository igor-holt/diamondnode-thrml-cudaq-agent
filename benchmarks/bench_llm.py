"""Ollama LLM inference benchmark (tok/s, p50 per-token latency, energy estimate).

Hits the local Ollama API on diamondnode. Models + prompt + num_predict are
configurable; default models are the ones known on the node.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

PROMPT = ("List the three most important principles of thermodynamics. "
          "Keep the answer under 40 words.")

# qwen2:0.5b is the default tracked model — llama3.2:3b fails to load when the
# fleet's grok llama-server holds the GTX 1650 VRAM (2.8 of 4 GB resident).
DEFAULT_MODELS = ["qwen2:0.5b", "llama3.2:3b"]
DEFAULT_TRIALS = 3
DEFAULT_NUM_PREDICT = 64


def ollama_stats(model: str, base_url: str = "http://127.0.0.1:11434",
                 num_predict: int = DEFAULT_NUM_PREDICT, timeout: float = 900.0) -> dict:
    """One generate call; return timing + token stats from the Ollama response.

    Handles both ms-prefixed (newer ollama) and ns-prefixed (older ollama)
    duration fields. Decode tok/s is computed from eval_duration; wall tok/s
    from the full request (includes model load + prompt eval + queueing).
    """
    payload = json.dumps({"model": model, "prompt": PROMPT, "stream": False,
                          "options": {"num_predict": num_predict}}).encode()
    req = urllib.request.Request(f"{base_url}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    dt = time.perf_counter() - t0

    def _sec(field: str, ms_field: str, ns_field: str) -> float | None:
        if data.get(ms_field) is not None:
            return data[ms_field] / 1000.0
        if data.get(ns_field) is not None:
            return data[ns_field] / 1e9
        return None

    n = int(data.get("eval_count", 0))
    eval_s = _sec("eval_duration", "eval_duration_ms", "eval_duration")
    prompt_s = _sec("prompt_eval_duration", "prompt_eval_duration_ms", "prompt_eval_duration")
    load_s = _sec("load_duration", "load_duration_ms", "load_duration")
    return {
        "model": model,
        "tokens": n,
        "wall_s": round(dt, 3),
        "eval_s": round(eval_s, 3) if eval_s is not None else None,
        "tok_per_s_decode": round(n / eval_s, 2) if eval_s and eval_s > 0 else 0.0,
        "tok_per_s_wall": round(n / dt, 2) if dt > 0 else 0.0,
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "prompt_s": round(prompt_s, 3) if prompt_s is not None else None,
        "load_s": round(load_s, 3) if load_s is not None else None,
    }


def run_llm_benchmark(models: list[str] | None = None, trials: int = DEFAULT_TRIALS,
                      base_url: str = "http://127.0.0.1:11434") -> dict:
    models = models or DEFAULT_MODELS
    results = {}
    for model in models:
        try:
            runs = [ollama_stats(model, base_url) for _ in range(trials)]
            dec = [r["tok_per_s_decode"] for r in runs if r["tok_per_s_decode"] > 0]
            wal = [r["tok_per_s_wall"] for r in runs if r["tok_per_s_wall"] > 0]
            lat = [r["eval_s"] / r["tokens"] for r in runs if r["eval_s"] and r["tokens"] > 0]
            loads = [r["load_s"] for r in runs if r["load_s"] is not None]
            results[model] = {
                "trials": trials,
                "status": "ok",
                "tok_per_s_decode_mean": round(statistics.mean(dec), 2) if dec else 0.0,
                "tok_per_s_wall_mean": round(statistics.mean(wal), 2) if wal else 0.0,
                "tok_per_s_decode_p50": round(statistics.median(dec), 2) if dec else 0.0,
                "ms_per_token_decode_p50": round(statistics.median(lat) * 1e3, 3) if lat else 0.0,
                "tokens_total": sum(r["tokens"] for r in runs),
                "load_s_mean": round(statistics.mean(loads), 3) if loads else None,
                # GPU power proxy: GTX 1650 draws ~50 W during decode -> J/tok
                "joules_per_token_est": round(50.0 / (statistics.mean(dec) if dec else 1.0), 3),
                "runs": runs,
            }
        except Exception as exc:
            results[model] = {"trials": 0, "status": f"error: {exc}", "runs": []}
    return {"backend": "ollama-" + base_url.split("//")[-1], "results": results}


if __name__ == "__main__":
    import sys
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = p.parse_args()
    print(json.dumps(run_llm_benchmark(args.models, args.trials), indent=2))
