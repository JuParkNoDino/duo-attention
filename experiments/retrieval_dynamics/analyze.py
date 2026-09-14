"""Plot temporal traces and summarize threshold sensitivity without pooling prompts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(rows, metric, epsilon):
    observed = [r for r in sorted(rows, key=lambda r: r["step"]) if r.get(metric) is not None]
    active = [r[metric] > epsilon for r in observed]
    eligible = [r for r in observed if r["eligible"]]
    consecutive = all(b["step"] == a["step"] + 1 for a, b in zip(observed, observed[1:]))
    longest, streak, events = 0, 0, 0
    previous = False
    for flag in active:
        events += int(flag and not previous)
        streak = 0 if flag else streak + 1
        longest = max(longest, streak)
        previous = flag
    return {
        "metric": metric, "epsilon": epsilon, "n_observed": len(observed),
        "n_eligible": len(eligible),
        "rho": float(np.mean(active)) if active else None,
        "rho_eligible": float(np.mean([r[metric] > epsilon for r in eligible])) if eligible else None,
        "event_runs": events if consecutive and observed else None,
        "longest_inactive_run": longest if consecutive and observed else None,
        "mean": float(np.mean([r[metric] for r in observed])) if observed else None,
        "p95": float(np.quantile([r[metric] for r in observed], .95)) if observed else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--metric", choices=["delta", "relative_delta", "g_far", "g_discarded", "kl"], default="delta")
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    p.add_argument("--plot-unit", choices=["kv", "query"], default="kv")
    p.add_argument("--plot-controls", action="store_true")
    a = p.parse_args()
    if any(not np.isfinite(e) or e < 0 for e in a.epsilons):
        p.error("epsilons must be finite and nonnegative")
    metadata = json.loads((a.run / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        p.error("Run incomplete; do not interpret a partial trajectory as a completed experiment")
    groups = defaultdict(list)
    with (a.run / "metrics.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            groups[r["sample_id"], r["layer"], r["unit"], r["head"], r["window"]].append(r)
    if not groups:
        p.error("No observations found")
    summary = []
    for key, rows in sorted(groups.items()):
        if len({r["step"] for r in rows}) != len(rows):
            p.error(f"Duplicate step in {key}")
        for epsilon in a.epsilons:
            summary.append({**dict(zip(["sample_id", "layer", "unit", "head", "window"], key)),
                            "retrieval": rows[0]["retrieval"], **summarize(rows, a.metric, epsilon)})
    path = a.run / f"summary_{a.metric}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotdir = a.run / f"plots_{a.metric}_{a.plot_unit}"
    plotdir.mkdir(exist_ok=True)
    index = []
    for key, rows in sorted(groups.items()):
        sample, layer, unit, head, window = key
        if unit != a.plot_unit or (not rows[0]["retrieval"] and not a.plot_controls):
            continue
        rows = sorted(rows, key=lambda r: r["step"])
        valid = [r for r in rows if r.get(a.metric) is not None]
        if not valid:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].plot([r["step"] for r in rows], [r["g_far"] for r in rows], label="G: outside local window")
        axes[0].plot([r["step"] for r in rows], [r["g_discarded"] for r in rows], label="Mass discarded (sinks retained)")
        axes[0].set(ylabel="Attention mass", ylim=(-.02, 1.02))
        axes[0].legend(fontsize=8)
        axes[1].plot([r["step"] for r in valid], [r[a.metric] for r in valid], marker="." if a.metric == "kl" else None)
        for epsilon in a.epsilons:
            axes[1].axhline(epsilon, linestyle="--", alpha=.4, label=f"epsilon={epsilon:g}")
        axes[1].set(xlabel="Decode step (0 predicts first continuation token)", ylabel=a.metric)
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{sample} | layer {layer}, {unit} head {head}, W={window} | retrieval={rows[0]['retrieval']}")
        fig.tight_layout()
        name = f"{hashlib.sha256(sample.encode()).hexdigest()[:12]}_L{layer}_{unit}{head}_W{window}.png"
        fig.savefig(plotdir / name, dpi=140)
        plt.close(fig)
        index.append({"sample_id": sample, "layer": layer, "head": head, "window": window, "file": name})
    (plotdir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Saved {path} and {len(index)} plots in {plotdir}")


if __name__ == "__main__":
    main()
