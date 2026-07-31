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
    """One generate call; return timing + token stats from the Ollama response."""
    payload = json.dumps({"model": model, "prompt": PROMPT, "stream": False,
                          "options": {"num_predict": num_predict}}).encode()
    req = urllib.request.Request(f"{base_url}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    dt = time.perf_counter() - t0
    n = int(data.get("eval_count", 0))
    # eval_duration_ms covers the decode window
    eval_s = data.get("eval_duration_ms", 0.0) / 1000.0 or dt
    return {
        "model": model,
        "tokens": n,
        "wall_s": round(dt, 3),
        "eval_s": round(eval_s, 3),
        "tok_per_s": round(n / eval_s, 2) if eval_s > 0 else 0.0,
        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
        "load_s": round((data.get("load_duration_ms", 0.0)) / 1000.0, 3),
    }


def run_llm_benchmark(models: list[str] | None = None, trials: int = DEFAULT_TRIALS,
                      base_url: str = "http://127.0.0.1:11434") -> dict:
    models = models or DEFAULT_MODELS
    results = {}
    for model in models:
        try:
            runs = [ollama_stats(model, base_url) for _ in range(trials)]
            tok_rates = [r["tok_per_s"] for r in runs if r["tok_per_s"] > 0]
            lat = [r["eval_s"] / r["tokens"] for r in runs if r["tokens"] > 0]
            results[model] = {
                "trials": trials,
                "status": "ok",
                "tok_per_s_mean": round(statistics.mean(tok_rates), 2) if tok_rates else 0.0,
                "tok_per_s_p50": round(statistics.median(tok_rates), 2) if tok_rates else 0.0,
                "ms_per_token_p50": round(statistics.median(lat) * 1e3, 3) if lat else 0.0,
                "tokens_total": sum(r["tokens"] for r in runs),
                "load_s_mean": round(statistics.mean(r["load_s"] for r in runs), 3),
                # GPU power proxy: GTX 1650 draws ~50 W during decode -> J/tok
                "joules_per_token_est": round(50.0 / (statistics.mean(tok_rates) if tok_rates else 1.0), 3),
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
