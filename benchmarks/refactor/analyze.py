"""Audit paired runs and plot runtime/work Dolan–More profiles."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    rows = {}
    for line in (args.run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["problem"], r["variant"], r["repeat"]] = r
    specs = manifest["problems"]
    repeats = manifest["repeats"]
    expected = {
        (p["id"], v, r)
        for p in specs
        for v in ["baseline", "candidate"]
        for r in range(repeats)
    }
    missing = sorted(expected - rows.keys())
    assert not missing, f"{len(missing)} missing runs"
    broken = [
        k
        for k in expected
        if rows[k]["result"]["status"]
        in ["harness_error", "worker_failed", "external_timeout"]
    ]
    assert not broken, f"{len(broken)} incomplete/invalid runs: {broken[:10]}"
    ignored = {"elapsed", "solve_time"}
    differences = []
    censored = []
    for p in specs:
        for rep in range(repeats):
            a, b = [rows[p["id"], v, rep]["result"] for v in ["baseline", "candidate"]]
            diff = [
                k
                for k in a.keys() | b.keys()
                if k not in ignored and a.get(k) != b.get(k)
            ]
            # Wall-clock censoring can stop two otherwise identical trajectories
            # at different iterations. Report these pairs separately, never exact.
            if a["status"] == "time_limit" or b["status"] == "time_limit":
                censored.append({"problem": p["id"], "repeat": rep, "fields": diff})
            elif diff:
                differences.append({"problem": p["id"], "repeat": rep, "fields": diff})
    timings = np.full((len(specs), 2, repeats), np.inf)
    iterations = np.full_like(timings, np.inf)
    counts = {}
    for i, p in enumerate(specs):
        for j, v in enumerate(["baseline", "candidate"]):
            results = [rows[p["id"], v, r]["result"] for r in range(repeats)]
            counts.setdefault(p["dataset"], {}).setdefault(v, Counter()).update(
                r["status"] for r in results
            )
            for rep, r in enumerate(results):
                if r.get("verified"):
                    timings[i, j, rep] = r["elapsed"]
                    iterations[i, j, rep] = max(1, r["iters"])
    medians = np.median(timings, axis=2)
    common = np.isfinite(medians).all(axis=1)
    ratios = medians[common, 1] / medians[common, 0]
    stable = common & (medians.min(axis=1) >= 0.01)
    stable_ratios = medians[stable, 1] / medians[stable, 0]
    noise = timings[stable, 0, 1] / timings[stable, 0, 0]
    thresholds = np.array([1.0, 1.01, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0])

    def profile(costs, grid):
        best = costs.min(axis=1)
        relative = np.full_like(costs, np.inf)
        np.divide(costs, best[:, None], out=relative, where=np.isfinite(best[:, None]))
        return (relative[:, :, None] <= grid[None, None, :]).mean(axis=0)

    curves = profile(medians, thresholds)
    baseline_repeat_curves = profile(timings[:, 0, :2], thresholds)
    rng = np.random.default_rng(20261016)
    bootstrap = (
        np.median(
            rng.choice(stable_ratios, (2000, len(stable_ratios)), replace=True), axis=1
        )
        if len(stable_ratios)
        else np.array([])
    )
    summary = {
        "problems": len(specs),
        "runs": len(expected),
        "repeats": repeats,
        "completed_pairs": len(specs) * repeats,
        "numerical_differences": differences,
        "time_censored_pairs": censored,
        "status_counts": counts,
        "verified_problems": {
            v: int(np.isfinite(medians[:, j]).sum())
            for j, v in enumerate(["baseline", "candidate"])
        },
        "common_verified": int(common.sum()),
        "median_candidate_over_baseline": float(np.median(ratios)),
        "over_10ms_count": int(stable.sum()),
        "over_10ms_median_ratio": float(np.median(stable_ratios))
        if len(stable_ratios)
        else None,
        "over_10ms_ratio_quantiles": np.quantile(
            stable_ratios, [0.1, 0.25, 0.5, 0.75, 0.9]
        ).tolist()
        if len(stable_ratios)
        else [],
        "baseline_repeat_ratio_quantiles": np.quantile(
            noise, [0.1, 0.25, 0.5, 0.75, 0.9]
        ).tolist()
        if len(noise)
        else [],
        "profile_thresholds": thresholds.tolist(),
        "runtime_profiles": curves.tolist(),
        "baseline_repeat_profiles": baseline_repeat_curves.tolist(),
        "over_10ms_median_ratio_bootstrap_95ci": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist()
        if len(bootstrap)
        else [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with gzip.open(args.output / "results.jsonl.gz", "wt") as f:
        for key in sorted(expected):
            f.write(json.dumps(rows[key]) + "\n")
    colors = ["#2456a6", "#db6d28"]
    labels = ["Before (c74a982)", "After refactor"]
    grid = np.unique(np.r_[np.linspace(1, 1.2, 201), np.geomspace(1.2, 10, 200)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for ax, costs, title in [
        (axes[0], medians, "Runtime"),
        (axes[1], np.median(iterations, axis=2), "Iterations"),
    ]:
        y = profile(costs, grid)
        for j in range(2):
            ax.step(
                grid,
                y[j],
                where="post",
                label=labels[j],
                color=colors[j],
                linestyle="-" if j == 0 else "--",
                linewidth=1.8,
            )
        ax.set_xscale("log")
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ratio to best build")
        ax.set_ylabel("Fraction of all problems")
        ax.set_title(title + " performance profile")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    ax = axes[2]
    bins = np.linspace(-0.5, 0.5, 40)
    if len(stable_ratios):
        ax.hist(
            np.log2(noise),
            bins=bins,
            alpha=0.4,
            label="Before: repeat 2 / repeat 1",
            color="#777777",
            density=True,
        )
        ax.hist(
            np.log2(stable_ratios),
            bins=bins,
            histtype="step",
            linewidth=2,
            label="After / before (medians)",
            color=colors[1],
            density=True,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log2 runtime ratio")
    ax.set_ylabel("Density")
    ax.set_title("Timing variation (solves ≥10 ms)")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"{len(specs)} problems · two serial repeats · one thread · presolve off",
        fontsize=12,
    )
    fig.savefig(args.output / "profiles.png", dpi=180)
    fig.savefig(args.output / "profiles.svg")
    # Individual collections, with all their failures retained in denominators.
    datasets = sorted({s["dataset"] for s in specs})
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        selected = np.array([s["dataset"] == dataset for s in specs])
        y = profile(medians[selected], grid)
        for j in range(2):
            ax.step(
                grid,
                y[j],
                where="post",
                label=labels[j],
                color=colors[j],
                linestyle="-" if j == 0 else "--",
            )
        ax.set_xscale("log")
        ax.set_xlim(1, 10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.set_title(f"{dataset} (n={selected.sum()})")
        ax.set_xlabel("Runtime ratio")
    axes.flat[len(datasets)].axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.savefig(args.output / "profiles-by-dataset.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
