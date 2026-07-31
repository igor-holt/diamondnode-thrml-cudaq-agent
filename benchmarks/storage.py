"""Baseline storage + regression detection for benchmark results."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_DIR = Path(__file__).resolve().parent.parent / "baselines"
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def resolve_path(root: dict, path: str):
    """Resolve a pipe-delimited path; list indexes like '-1' select from the end."""
    value = root
    for key in path.split("|"):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.lstrip("-").isdigit():
            idx = int(key)
            if idx < 0:
                idx += len(value)
            if 0 <= idx < len(value):
                value = value[idx]
            else:
                return None
        else:
            return None
    return value


class BaselineStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self._load() if self.path.exists() else {}

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"corrupt baseline file {self.path}: {exc}") from exc

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def set_metric(self, suite: str, metric: str, value: float, unit: str = "",
                   env: str = "") -> None:
        self.data.setdefault(suite, {})[metric] = {
            "value": value,
            "unit": unit,
            "env": env,
        }
        self.save()

    def metric(self, suite: str, metric: str) -> dict | None:
        return self.data.get(suite, {}).get(metric)

    def update_from(self, result: dict, suite: str, metrics: dict[str, str],
                    env: str = "") -> dict[str, Any]:
        """Record chosen metrics from a result dict; return the recorded rows."""
        recorded = {}
        for metric, path in metrics.items():
            value = resolve_path(result, path)
            if value is None or not isinstance(value, (int, float)):
                continue
            self.set_metric(suite, metric, float(value), unit="", env=env)
            recorded[metric] = float(value)
        return recorded

    def compare(self, suite: str, result: dict, metrics: dict[str, str],
                tolerances: dict[str, float]) -> list[dict]:
        """Compare a new result against stored baseline. Higher-is-better unless
        `tolerances` marks a metric `{"type": "lower"}`. Returns rows with
        delta %, threshold breach info. Tolerance is a relative fraction
        (e.g. 0.10 = 10%)."""
        rows = []
        for metric, path in metrics.items():
            value = resolve_path(result, path)
            base = self.metric(suite, metric)
            if value is None or not isinstance(value, (int, float)):
                continue
            if base is None:
                rows.append({"metric": metric, "value": value, "baseline": None,
                             "delta_pct": None, "status": "no-baseline"})
                continue
            base_val = base["value"]
            spec = tolerances.get(metric, {"type": "higher"})
            better = value > base_val if spec.get("type", "higher") == "higher" else value < base_val
            delta_pct = (value - base_val) / abs(base_val) * 100.0 if base_val else 0.0
            limit = spec.get("limit", 0.10)
            breached = (not better) and abs(delta_pct) > limit * 100.0
            rows.append({
                "metric": metric,
                "value": value,
                "baseline": base_val,
                "delta_pct": round(delta_pct, 2),
                "status": "regression" if breached else "ok",
            })
        return rows


def summarize(rows: list[dict]) -> dict:
    regressions = [r for r in rows if r["status"] == "regression"]
    return {
        "checked": len(rows),
        "regressions": len(regressions),
        "no_baseline": sum(1 for r in rows if r["status"] == "no-baseline"),
        "ok": sum(1 for r in rows if r["status"] == "ok"),
        "details": rows,
    }


def exit_code_for(summary: dict) -> int:
    return 1 if summary["regressions"] else 0


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
    sys.exit(0)
