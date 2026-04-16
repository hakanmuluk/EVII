#!/usr/bin/env python3
"""
Other-metrics correlation runner.

Computes four simple scalar metrics from ``token_metrics_trace`` for every
per-example JSON, normalises each metric to [0, 1] within each version, and
evaluates weighted-binned Pearson correlation against per-example accuracy.

Metrics
-------
avg_log_prob        mean of per-token log-probabilities
avg_self_certainty  mean of per-token self-certainty scores
avg_neg_entropy     negative mean of per-token entropies (nats)
avg_neg_perplexity  negative mean of per-token perplexities (exp(H), nats)

No hyperparameter grid is needed — each metric is computed deterministically.

Usage
-----
    python analyze_other_metrics_correlation.py \\
        --json-dirs   dir1,dir2,...,dir6 \\
        --results-jsons res1.json,...,res6.json \\
        --version-names modelA_ds1,...,modelB_ds3 \\
        --output-dir  my_run \\
        --bin-width   0.05
"""

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRIC_NAMES: List[str] = [
    "avg_log_prob",
    "avg_self_certainty",
    "avg_neg_entropy",
    "avg_neg_perplexity",
]

METRIC_LABELS: Dict[str, str] = {
    "avg_log_prob":        "Avg log-probability",
    "avg_self_certainty":  "Avg self-certainty",
    "avg_neg_entropy":     "Avg negative entropy (nats)",
    "avg_neg_perplexity":  "Avg negative perplexity (exp H, nats)",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExampleRecord:
    example_id: str
    source_path: Path
    correct: int
    avg_log_prob: float
    avg_self_certainty: float
    avg_neg_entropy: float
    avg_neg_perplexity: float

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(results_json_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (by_example_id, by_filename_stem) ground-truth label dicts."""
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_example = data["per_example"]
    by_example_id: Dict[str, int] = {}
    by_filename_stem: Dict[str, int] = {}

    for key, item in per_example.items():
        ex_id = item["example_id"]
        correct = int(item["correct"])
        by_example_id[ex_id] = correct
        by_filename_stem[Path(key).stem] = correct

    return by_example_id, by_filename_stem


def load_example_record(json_path: Path, correct: int) -> ExampleRecord:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    example_id = data["example"]["example_id"]
    tmt = data["token_metrics_trace"]

    log_probs       = np.asarray(tmt["log_probs"],        dtype=float)
    entropies       = np.asarray(tmt["entropies"],         dtype=float)
    self_certainties = np.asarray(tmt["self_certainties"], dtype=float)

    avg_log_prob       = float(np.mean(log_probs))
    avg_self_certainty = float(np.mean(self_certainties))
    avg_neg_entropy    = float(-np.mean(entropies))
    # perplexity = exp(H) since entropies are in nats
    avg_neg_perplexity = float(-np.mean(np.exp(entropies)))

    return ExampleRecord(
        example_id=example_id,
        source_path=json_path,
        correct=int(correct),
        avg_log_prob=avg_log_prob,
        avg_self_certainty=avg_self_certainty,
        avg_neg_entropy=avg_neg_entropy,
        avg_neg_perplexity=avg_neg_perplexity,
    )


def load_records(
    json_dir: Path,
    results_json: Path,
    skip_missing: bool,
) -> Tuple[List[ExampleRecord], List[Tuple[str, str]]]:
    by_example_id, by_filename_stem = load_results(results_json)

    records: List[ExampleRecord] = []
    skipped: List[Tuple[str, str]] = []

    for json_path in sorted(json_dir.glob("*.json")):
        if json_path.resolve() == results_json.resolve():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "example" not in data or "token_metrics_trace" not in data:
                skipped.append((str(json_path), "missing 'example' or 'token_metrics_trace'"))
                continue

            example_id = data["example"]["example_id"]

            if example_id in by_example_id:
                correct = by_example_id[example_id]
            elif json_path.stem in by_filename_stem:
                correct = by_filename_stem[json_path.stem]
            else:
                msg = f"missing result label for {example_id}"
                if skip_missing:
                    skipped.append((str(json_path), msg))
                    continue
                raise KeyError(msg)

            record = load_example_record(json_path=json_path, correct=correct)
            records.append(record)

        except Exception as e:
            skipped.append((str(json_path), str(e)))

    if len(records) == 0:
        raise RuntimeError(
            f"No usable per-example JSON files loaded from {json_dir}"
        )

    return records, skipped

# ---------------------------------------------------------------------------
# Metric computation + normalisation
# ---------------------------------------------------------------------------

def _raw_metric_array(records: List[ExampleRecord], metric_name: str) -> np.ndarray:
    return np.array([getattr(r, metric_name) for r in records], dtype=float)


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]. Degenerate (flat) arrays become all 0.5."""
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi == lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def compute_normalized_metrics(
    records: List[ExampleRecord],
) -> Dict[str, np.ndarray]:
    """Return {metric_name: normalized_array} for all METRIC_NAMES."""
    result: Dict[str, np.ndarray] = {}
    for mname in METRIC_NAMES:
        raw = _raw_metric_array(records, mname)
        result[mname] = minmax_normalize(raw)
    return result

# ---------------------------------------------------------------------------
# Binning + weighted Pearson (self-contained, mirrors existing file)
# ---------------------------------------------------------------------------

def make_ratio_bins(bin_width: float = 0.05, max_ratio: float = 1.0) -> np.ndarray:
    max_ratio = max(float(max_ratio), 1.0)
    edges = np.arange(0.0, max_ratio + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([0.0, max_ratio + bin_width], dtype=float)
    if edges[-1] < max_ratio:
        edges = np.append(edges, max_ratio)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    return edges


def compute_binned_accuracy(
    metric_values: np.ndarray,
    correct_values: np.ndarray,
    bin_width: float = 0.05,
) -> List[dict]:
    metric_values  = np.asarray(metric_values,  dtype=float)
    correct_values = np.asarray(correct_values, dtype=float)

    valid = np.isfinite(metric_values) & np.isfinite(correct_values)
    metric_values  = metric_values[valid]
    correct_values = correct_values[valid]

    if len(metric_values) == 0:
        return []

    edges = make_ratio_bins(
        bin_width=bin_width,
        max_ratio=max(1.0, float(np.nanmax(metric_values))),
    )

    bin_ids = np.digitize(metric_values, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        left   = float(edges[b])
        right  = float(edges[b + 1])
        center = 0.5 * (left + right)
        rows.append({
            "bin_left":         left,
            "bin_right":        right,
            "bin_center":       center,
            "count":            int(mask.sum()),
            "mean_accuracy":    float(np.mean(correct_values[mask])),
            "mean_metric_value": float(np.mean(metric_values[mask])),
        })
    return rows


def weighted_binned_pearson(binned_rows: List[dict]) -> float:
    if len(binned_rows) < 2:
        return float("nan")

    x = np.array([r["bin_center"]    for r in binned_rows], dtype=float)
    y = np.array([r["mean_accuracy"] for r in binned_rows], dtype=float)
    w = np.array([r["count"]         for r in binned_rows], dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[valid], y[valid], w[valid]

    if len(x) < 2:
        return float("nan")

    w_sum = np.sum(w)
    if w_sum <= 0:
        return float("nan")

    mx = np.sum(w * x) / w_sum
    my = np.sum(w * y) / w_sum
    vx = np.sum(w * (x - mx) ** 2) / w_sum
    vy = np.sum(w * (y - my) ** 2) / w_sum

    if vx <= 0 or vy <= 0:
        return float("nan")

    cov  = np.sum(w * (x - mx) * (y - my)) / w_sum
    corr = cov / math.sqrt(vx * vy)
    return float(max(min(corr, 1.0), -1.0))

# ---------------------------------------------------------------------------
# Per-metric evaluation across all versions
# ---------------------------------------------------------------------------

def evaluate_metric(
    metric_name: str,
    versions: List[Tuple[str, List[ExampleRecord]]],
    bin_width: float,
) -> dict:
    """
    For one metric, compute normalised values for each version, run binned
    correlation, and return a summary dict.
    """
    version_correlations: Dict[str, float]   = {}
    version_nonempty_bins: Dict[str, int]    = {}
    binned_rows_v0: List[dict]               = []

    for idx, (vname, records) in enumerate(versions):
        norm_metrics = compute_normalized_metrics(records)
        norm_vals    = norm_metrics[metric_name]
        correct_vals = np.array([r.correct for r in records], dtype=float)

        binned = compute_binned_accuracy(norm_vals, correct_vals, bin_width=bin_width)
        corr   = weighted_binned_pearson(binned)

        version_correlations[vname]  = corr
        version_nonempty_bins[vname] = len(binned)

        if idx == 0:
            binned_rows_v0 = binned

    finite_corrs = [c for c in version_correlations.values() if not math.isnan(c)]
    sum_corr = float(sum(finite_corrs)) if finite_corrs else float("nan")

    return {
        "metric_name":         metric_name,
        "version_correlations":  version_correlations,
        "version_nonempty_bins": version_nonempty_bins,
        "sum_corr":              sum_corr,
        "binned_rows_v0":        binned_rows_v0,
    }

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_for_path(value: str) -> str:
    s = str(value).replace("-", "neg_").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9._]+", "_", s).strip("_") or "x"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_binned_curve(
    binned_rows: List[dict],
    output_path: Path,
    title: str,
    metric_label: str,
) -> None:
    if len(binned_rows) == 0:
        return

    x      = np.array([r["bin_center"]    for r in binned_rows], dtype=float)
    y      = np.array([r["mean_accuracy"] for r in binned_rows], dtype=float)
    counts = np.array([r["count"]         for r in binned_rows], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel(f"{metric_label} (normalised) bin center")
    plt.ylabel("Average accuracy in bin")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.title(title)

    for xi, yi, c in zip(x, y, counts):
        plt.annotate(
            f"n={int(c)}", (xi, yi),
            textcoords="offset points", xytext=(0, 6), ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def collect_fieldnames(rows: List[dict]) -> List[str]:
    seen: set = set()
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def save_csv(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = collect_fieldnames(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ---------------------------------------------------------------------------
# Saving per-metric artefacts
# ---------------------------------------------------------------------------

def save_metric_results(
    result: dict,
    output_dir: Path,
    first_version_name: str,
) -> None:
    mname  = result["metric_name"]
    mlabel = METRIC_LABELS.get(mname, mname)
    metric_dir = output_dir / mname
    safe_mkdir(metric_dir)

    corr_v0     = result["version_correlations"].get(first_version_name, float("nan"))
    corr_v0_str = f"{corr_v0:.4f}" if not math.isnan(corr_v0) else "NaN"
    title = (
        f"{mlabel} vs accuracy  ({first_version_name})\n"
        f"weighted binned Pearson = {corr_v0_str}"
    )

    plot_binned_curve(
        result["binned_rows_v0"],
        metric_dir / "binned_plot_v0.png",
        title=title,
        metric_label=mlabel,
    )

    save_csv(result["binned_rows_v0"], metric_dir / "binned_metrics_v0.csv")

    serialisable = {k: v for k, v in result.items() if k != "binned_rows_v0"}
    with open(metric_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False, default=_json_default)

# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

def save_summary(
    all_results: List[dict],
    output_dir: Path,
    version_names: List[str],
) -> List[dict]:
    def sort_key(r: dict) -> float:
        v = r["sum_corr"]
        return v if not math.isnan(v) else -float("inf")

    ranked = sorted(all_results, key=sort_key, reverse=True)

    summary_rows: List[dict] = []
    for r in ranked:
        row: dict = {"metric_name": r["metric_name"]}
        row["sum_corr"] = r["sum_corr"]
        for vname in version_names:
            row[f"corr_{vname}"]         = r["version_correlations"].get(vname, float("nan"))
            row[f"nonempty_bins_{vname}"] = r["version_nonempty_bins"].get(vname, 0)
        summary_rows.append(row)

    save_csv(summary_rows, output_dir / "all_metrics_ranked.csv")

    all_sum_corrs = [r["sum_corr"] for r in all_results if not math.isnan(r["sum_corr"])]
    avg_sum_corr  = float(np.mean(all_sum_corrs)) if all_sum_corrs else float("nan")

    summary_json = {
        "metrics": METRIC_NAMES,
        "versions": version_names,
        "avg_sum_corr_all": avg_sum_corr,
        "ranked_metrics": [
            {k: v for k, v in r.items() if k != "binned_rows_v0"}
            for r in ranked
        ],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=_json_default)

    return ranked


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def print_metric_result(result: dict, version_names: List[str]) -> None:
    mname = result["metric_name"]
    corr_strs = []
    for vname in version_names:
        v = result["version_correlations"].get(vname, float("nan"))
        corr_strs.append(
            f"{vname}:{v:.4f}" if not math.isnan(v) else f"{vname}:NaN"
        )
    sc = result["sum_corr"]
    sc_str = f"{sc:.4f}" if not math.isnan(sc) else "NaN"
    print(f"  {mname:<24}  sum_corr={sc_str}  [{', '.join(corr_strs)}]")

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute avg log-prob, avg self-certainty, avg neg-entropy, and "
            "avg neg-perplexity from token_metrics_trace JSON files, normalise "
            "each to [0,1], and evaluate weighted-binned Pearson correlation "
            "against per-example accuracy across one or more versions."
        )
    )

    parser.add_argument(
        "--json-dirs", type=str, required=True,
        help="Comma-separated list of per-example JSON directories, one per version",
    )
    parser.add_argument(
        "--results-jsons", type=str, required=True,
        help="Comma-separated list of evaluation_results.json paths, one per version",
    )
    parser.add_argument(
        "--version-names", type=str, default=None,
        help=(
            "Optional comma-separated version names. "
            "Defaults to v0,v1,... if not supplied."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path for the output folder",
    )
    parser.add_argument(
        "--bin-width", type=float, default=0.05,
        help="Bin width for normalised-metric binning (default: 0.05)",
    )
    parser.add_argument(
        "--skip-missing-results", action="store_true",
        help="Skip per-example JSONs not found in the results JSON",
    )

    args = parser.parse_args()

    json_dirs     = [Path(p.strip()) for p in args.json_dirs.split(",")     if p.strip()]
    results_jsons = [Path(p.strip()) for p in args.results_jsons.split(",") if p.strip()]

    if len(json_dirs) != len(results_jsons):
        parser.error(
            f"--json-dirs ({len(json_dirs)}) and --results-jsons "
            f"({len(results_jsons)}) must have the same number of entries."
        )

    if args.version_names:
        version_names = [n.strip() for n in args.version_names.split(",") if n.strip()]
        if len(version_names) != len(json_dirs):
            parser.error(
                f"--version-names ({len(version_names)}) must match the number "
                f"of versions ({len(json_dirs)})."
            )
    else:
        version_names = [f"v{i}" for i in range(len(json_dirs))]

    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    print("=" * 72)
    print("Other-metrics correlation analysis")
    print(f"Versions ({len(version_names)}): {version_names}")
    print(f"Metrics:   {METRIC_NAMES}")
    print(f"Bin width: {args.bin_width}")
    print("=" * 72)

    # Load all versions
    versions: List[Tuple[str, List[ExampleRecord]]] = []
    for vname, jdir, rjson in zip(version_names, json_dirs, results_jsons):
        print(f"Loading version '{vname}' from {jdir} …")
        records, skipped = load_records(
            jdir, rjson, skip_missing=args.skip_missing_results
        )
        print(f"  Loaded {len(records)} examples, skipped {len(skipped)} files")
        if skipped:
            for path, reason in skipped[:5]:
                print(f"    skip: {Path(path).name} — {reason}")
            if len(skipped) > 5:
                print(f"    … and {len(skipped) - 5} more")
        versions.append((vname, records))

    print()

    # Evaluate all 4 metrics
    all_results: List[dict] = []
    for mname in METRIC_NAMES:
        result = evaluate_metric(mname, versions, bin_width=args.bin_width)
        all_results.append(result)
        save_metric_results(result, output_dir, first_version_name=version_names[0])

    print("Correlations per metric:")
    for result in all_results:
        print_metric_result(result, version_names)

    print()
    print("Saving summary …")
    ranked = save_summary(all_results, output_dir, version_names)

    print("\nRanking by sum_corr (best first):")
    for i, r in enumerate(ranked, 1):
        sc = r["sum_corr"]
        sc_str = f"{sc:.4f}" if not math.isnan(sc) else "NaN"
        print(f"  {i}. {r['metric_name']:<24}  sum_corr = {sc_str}")

    print("\nOutput written to:", output_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
