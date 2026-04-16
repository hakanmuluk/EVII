#!/usr/bin/env python3
"""
Static correlation runner aligned with metric_definition_dynamic.py.

This runner evaluates the metric defined in metric_definition_dynamic.py
against per-example accuracy using weighted-binned Pearson correlation.

Important alignment notes
-------------------------
The current dynamic metric expects:

    compute_metric_for_example(record, offset, **params)

and uses adaptive-k parameter names such as:
    adaptive_k_cp_prob_threshold
    adaptive_k_drop_ratio
    adaptive_k_max_search_tokens
    adaptive_k_smooth
    adaptive_k_prior_mean
    adaptive_k_prior_kappa
    adaptive_k_beta_concentration
    adaptive_k_pre_window
    adaptive_k_post_window
    adaptive_k_clip_eps

The metric file currently hardcodes:
    k_min = 0.05 * total_tokens
    expected_run_length = 0.10 * total_tokens
    k_max = 0.15 * total_tokens

so this runner does NOT expose k-triples, because they are not used by the
current metric_definition_dynamic.py.
"""

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metric_definition_dynamic as dynamic_metric

# ---------------------------------------------------------------------------
# Success thresholds
# ---------------------------------------------------------------------------
SUCCESS_MIN_CORR = 0.3
SUCCESS_HIGH_CORR = 0.4
SUCCESS_HIGH_COUNT = 5
SUCCESS_MIN_NONEMPTY_BINS = 20

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExampleRecord:
    example_id: str
    source_path: Path
    correct: int
    js_matrix: np.ndarray
    token_texts: list
    noimage_js: np.ndarray
    total_tokens: int

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def extract_noimage_js(data):
    noimg = data.get("noimage_comparison_trace")
    if noimg is None:
        raise KeyError("Missing 'noimage_comparison_trace'")

    if isinstance(noimg, dict) and "js_per_token" in noimg:
        return np.asarray(noimg["js_per_token"], dtype=float)

    if isinstance(noimg, dict):
        for key in ["per_token", "token_js", "tokens", "trace", "entries", "comparisons"]:
            if key in noimg and isinstance(noimg[key], list):
                items = noimg[key]
                js_vals = []
                if len(items) > 0 and isinstance(items[0], dict):
                    if "token_index" in items[0]:
                        items = sorted(items, key=lambda z: z["token_index"])
                    for item in items:
                        if "js_divergence" in item:
                            js_vals.append(float(item["js_divergence"]))
                        elif "js" in item:
                            js_vals.append(float(item["js"]))
                        else:
                            raise KeyError(f"Could not find js field in noimage item under '{key}'")
                    return np.asarray(js_vals, dtype=float)

    if isinstance(noimg, list):
        if len(noimg) == 0:
            return np.asarray([], dtype=float)
        if isinstance(noimg[0], (float, int)):
            return np.asarray(noimg, dtype=float)
        if isinstance(noimg[0], dict):
            items = noimg
            if "token_index" in items[0]:
                items = sorted(items, key=lambda z: z["token_index"])
            js_vals = []
            for item in items:
                if "js_divergence" in item:
                    js_vals.append(float(item["js_divergence"]))
                elif "js" in item:
                    js_vals.append(float(item["js"]))
                else:
                    raise KeyError("Could not find js field in noimage_comparison_trace list item")
            return np.asarray(js_vals, dtype=float)

    raise KeyError("Unsupported 'noimage_comparison_trace' format")


def load_results(results_json_path: Path):
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_example = data["per_example"]
    by_example_id = {}
    by_filename_stem = {}

    for key, item in per_example.items():
        ex_id = item["example_id"]
        correct = int(item["correct"])
        by_example_id[ex_id] = correct
        by_filename_stem[Path(key).stem] = correct

    return by_example_id, by_filename_stem


def load_example_record(json_path: Path, correct: int):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    example_id = data["example"]["example_id"]
    js_matrix = np.asarray(data["js_trace"]["js_matrix"], dtype=float)
    token_texts = data["js_trace"].get("token_texts", [])
    noimage_js = extract_noimage_js(data)
    _, total_tokens = js_matrix.shape

    return ExampleRecord(
        example_id=example_id,
        source_path=json_path,
        correct=int(correct),
        js_matrix=js_matrix,
        token_texts=token_texts,
        noimage_js=noimage_js,
        total_tokens=total_tokens,
    )


def load_records(json_dir: Path, results_json: Path, skip_missing: bool):
    by_example_id, by_filename_stem = load_results(results_json)

    records = []
    skipped = []

    for json_path in sorted(json_dir.glob("*.json")):
        if json_path.resolve() == results_json.resolve():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "example" not in data or "js_trace" not in data:
                skipped.append((str(json_path), "not a per-example trace json"))
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
        raise RuntimeError(f"No usable per-example JSON files were loaded from {json_dir}")

    return records, skipped

# ---------------------------------------------------------------------------
# Binning + weighted Pearson
# ---------------------------------------------------------------------------

def make_ratio_bins(bin_width=0.15, max_ratio=1.0):
    max_ratio = max(float(max_ratio), 1.0)
    edges = np.arange(0.0, max_ratio + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([0.0, max_ratio + bin_width], dtype=float)
    if edges[-1] < max_ratio:
        edges = np.append(edges, max_ratio)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    return edges


def compute_binned_accuracy(metric_values, correct_values, bin_width=0.15):
    metric_values = np.asarray(metric_values, dtype=float)
    correct_values = np.asarray(correct_values, dtype=float)

    valid = np.isfinite(metric_values) & np.isfinite(correct_values)
    metric_values = metric_values[valid]
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

        left = float(edges[b])
        right = float(edges[b + 1])
        center = 0.5 * (left + right)

        rows.append({
            "bin_left": left,
            "bin_right": right,
            "bin_center": center,
            "count": int(mask.sum()),
            "mean_accuracy": float(np.mean(correct_values[mask])),
            "mean_metric_value": float(np.mean(metric_values[mask])),
        })

    return rows


def weighted_binned_pearson(binned_rows):
    if len(binned_rows) < 2:
        return float("nan")

    x = np.array([r["bin_center"] for r in binned_rows], dtype=float)
    y = np.array([r["mean_accuracy"] for r in binned_rows], dtype=float)
    w = np.array([r["count"] for r in binned_rows], dtype=float)

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

    cov = np.sum(w * (x - mx) * (y - my)) / w_sum
    corr = cov / math.sqrt(vx * vy)
    corr = max(min(corr, 1.0), -1.0)
    return float(corr)

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_for_path(value):
    s = f"{value}"
    s = s.replace("-", "neg_").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9._]+", "_", s).strip("_") or "x"


import hashlib
import json

def combo_folder_name(params):
    keys = sorted(params.keys())

    short_parts = []
    for key in keys:
        value = sanitize_for_path(params[key])

        short_key_map = {
            "offset": "off",
            "ratio_bin_width": "bin",
            "adaptive_k_max_search_tokens": "mst",
            "adaptive_k_smooth": "sm",
            "adaptive_k_prior_mean": "pm",
            "adaptive_k_prior_kappa": "pk",
            "adaptive_k_beta_concentration": "bc",
            "adaptive_k_cp_prob_threshold": "cp",
            "adaptive_k_pre_window": "pre",
            "adaptive_k_post_window": "post",
            "adaptive_k_drop_ratio": "dr",
            "adaptive_k_clip_eps": "eps",
        }

        short_key = short_key_map.get(key, key[:4])
        short_parts.append(f"{short_key}_{value}")

    base = "__".join(short_parts)

    # add stable hash so names stay unique even if shortened later
    digest = hashlib.md5(
        json.dumps({k: params[k] for k in keys}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    # keep directory names comfortably short
    if len(base) > 120:
        base = base[:120]

    return f"{base}__{digest}"

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_binned_curve(binned_rows, output_path: Path, title: str):
    if len(binned_rows) == 0:
        return

    x = np.array([r["bin_center"] for r in binned_rows], dtype=float)
    y = np.array([r["mean_accuracy"] for r in binned_rows], dtype=float)
    counts = np.array([r["count"] for r in binned_rows], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel(f"{dynamic_metric.METRIC_NAME} bin center")
    plt.ylabel("Average accuracy in bin")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.title(title)

    for xi, yi, c in zip(x, y, counts):
        plt.annotate(
            f"n={int(c)}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def collect_fieldnames(rows):
    seen = set()
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def save_csv(rows, csv_path: Path):
    if len(rows) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            pass
        return

    fieldnames = collect_fieldnames(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_bool_list(s):
    if s is None or str(s).strip() == "":
        return []
    out = []
    for x in str(s).split(","):
        v = x.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif v in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Could not parse boolean value: {x}")
    return out


def parse_optional_float_list(s):
    """
    Parses comma-separated values where each value is either:
      - a float
      - 'none' / 'null'
    """
    if s is None or str(s).strip() == "":
        return [None]

    out = []
    for x in str(s).split(","):
        v = x.strip().lower()
        if v in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(x.strip()))
    return out

# ---------------------------------------------------------------------------
# Core evaluation for one parameter combination across all versions
# ---------------------------------------------------------------------------

def evaluate_combination(versions, params):
    bin_width = float(params.get("ratio_bin_width", 0.15))

    version_correlations = {}
    version_k_stats = {}
    version_nonempty_bins = {}
    binned_rows_v0 = []

    for idx, (vname, records) in enumerate(versions):
        metric_vals = []
        correct_vals = []
        chosen_ks = []

        for rec in records:
            row = dynamic_metric.compute_metric_for_example(record=rec, **params)
            metric_vals.append(row["metric_value"])
            correct_vals.append(row["correct"])
            chosen_ks.append(row["chosen_k"])

        binned = compute_binned_accuracy(metric_vals, correct_vals, bin_width=bin_width)
        corr = weighted_binned_pearson(binned)

        version_correlations[vname] = corr
        version_nonempty_bins[vname] = len(binned)

        ks_arr = np.asarray(chosen_ks, dtype=float)
        version_k_stats[vname] = {
            "k_mean": float(np.mean(ks_arr)) if len(ks_arr) > 0 else float("nan"),
            "k_std": float(np.std(ks_arr)) if len(ks_arr) > 0 else float("nan"),
            "nonempty_bins": len(binned),
        }

        if idx == 0:
            binned_rows_v0 = binned

    corrs = [v for v in version_correlations.values()]
    finite_corrs = [c for c in corrs if not math.isnan(c)]

    n_above_high = sum(1 for c in finite_corrs if c >= SUCCESS_HIGH_CORR)
    all_above_min = (
        len(finite_corrs) == len(corrs)
        and all(c >= SUCCESS_MIN_CORR for c in finite_corrs)
    )
    all_enough_bins = all(
        version_nonempty_bins.get(vname, 0) >= SUCCESS_MIN_NONEMPTY_BINS
        for vname, _ in versions
    )
    successful = (n_above_high >= SUCCESS_HIGH_COUNT) and all_above_min and all_enough_bins
    sum_corr = float(sum(finite_corrs)) if finite_corrs else float("nan")

    return {
        "params": dict(params),
        "version_correlations": version_correlations,
        "version_k_stats": version_k_stats,
        "version_nonempty_bins": version_nonempty_bins,
        "sum_corr": sum_corr,
        "successful": successful,
        "binned_rows_v0": binned_rows_v0,
    }

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

def build_param_grid(
    offsets,
    adaptive_k_max_search_tokens,
    adaptive_k_smooth,
    adaptive_k_prior_mean,
    adaptive_k_prior_kappa,
    adaptive_k_beta_concentration,
    adaptive_k_cp_prob_threshold,
    adaptive_k_pre_window,
    adaptive_k_post_window,
    adaptive_k_drop_ratio,
    adaptive_k_clip_eps,
    ratio_bin_widths,
):
    grid_spec = {
        "offset": offsets,
        "adaptive_k_max_search_tokens": adaptive_k_max_search_tokens,
        "adaptive_k_smooth": adaptive_k_smooth,
        "adaptive_k_prior_mean": adaptive_k_prior_mean,
        "adaptive_k_prior_kappa": adaptive_k_prior_kappa,
        "adaptive_k_beta_concentration": adaptive_k_beta_concentration,
        "adaptive_k_cp_prob_threshold": adaptive_k_cp_prob_threshold,
        "adaptive_k_pre_window": adaptive_k_pre_window,
        "adaptive_k_post_window": adaptive_k_post_window,
        "adaptive_k_drop_ratio": adaptive_k_drop_ratio,
        "adaptive_k_clip_eps": adaptive_k_clip_eps,
        "ratio_bin_width": ratio_bin_widths,
    }

    keys = sorted(grid_spec.keys())
    values = [grid_spec[k] for k in keys]

    param_grid = []
    for combo in product(*values):
        param_grid.append(dict(zip(keys, combo)))
    return param_grid

# ---------------------------------------------------------------------------
# Save artefacts for a single combination
# ---------------------------------------------------------------------------

def save_combination(result, combo_dir: Path, first_version_name: str):
    safe_mkdir(combo_dir)

    corr_v0 = result["version_correlations"].get(first_version_name, float("nan"))
    corr_v0_str = f"{corr_v0:.4f}" if not math.isnan(corr_v0) else "NaN"

    title = (
        f"{dynamic_metric.METRIC_NAME} vs accuracy ({first_version_name})\n"
        f"weighted binned Pearson = {corr_v0_str}"
    )

    plot_binned_curve(
        result["binned_rows_v0"],
        combo_dir / "binned_plot_v0.png",
        title,
    )

    save_csv(result["binned_rows_v0"], combo_dir / "binned_metrics_v0.csv")

    serialisable = {k: v for k, v in result.items() if k != "binned_rows_v0"}
    with open(combo_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False, default=_json_default)

    return combo_dir / "binned_plot_v0.png"


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
# Top-level summary
# ---------------------------------------------------------------------------

def save_top_level_summary(all_results, output_dir: Path,
                           best_successful_plot: Optional[Path],
                           version_names: list):
    successful = [r for r in all_results if r["successful"]]
    unsuccessful = [r for r in all_results if not r["successful"]]

    def sort_key(r):
        v = r["sum_corr"]
        return v if not math.isnan(v) else -float("inf")

    successful.sort(key=sort_key, reverse=True)
    unsuccessful.sort(key=sort_key, reverse=True)

    ranked = successful + unsuccessful

    summary_rows = []
    for r in ranked:
        row = dict(r["params"])
        row["successful"] = r["successful"]
        row["sum_corr"] = r["sum_corr"]

        for vname in version_names:
            row[f"corr_{vname}"] = r["version_correlations"].get(vname, float("nan"))

        for vname in version_names:
            k_stats = r["version_k_stats"].get(vname, {})
            row[f"k_mean_{vname}"] = k_stats.get("k_mean", float("nan"))
            row[f"k_std_{vname}"] = k_stats.get("k_std", float("nan"))
            row[f"nonempty_bins_{vname}"] = k_stats.get("nonempty_bins", float("nan"))

        summary_rows.append(row)

    save_csv(summary_rows, output_dir / "all_combinations_ranked.csv")

    all_sum_corrs = [r["sum_corr"] for r in all_results if not math.isnan(r["sum_corr"])]
    avg_sum_corr = float(np.mean(all_sum_corrs)) if all_sum_corrs else float("nan")

    summary_json = {
        "metric_name": dynamic_metric.METRIC_NAME,
        "total_combinations": len(all_results),
        "successful_count": len(successful),
        "unsuccessful_count": len(unsuccessful),
        "avg_sum_corr_all": avg_sum_corr,
        "success_criteria": {
            "min_corr_all_versions": SUCCESS_MIN_CORR,
            "high_corr_threshold": SUCCESS_HIGH_CORR,
            "min_versions_above_high": SUCCESS_HIGH_COUNT,
            "min_nonempty_bins_per_version": SUCCESS_MIN_NONEMPTY_BINS,
        },
        "ranked_combinations": [
            {k: v for k, v in r.items() if k != "binned_rows_v0"}
            for r in ranked
        ],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=_json_default)

    if best_successful_plot is not None and best_successful_plot.exists():
        shutil.copy2(best_successful_plot, output_dir / "best_successful_binned_plot.png")

    return ranked

# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

def print_combination(result, idx, total, version_names):
    status = "SUCCESSFUL" if result["successful"] else "UNSUCCESSFUL"

    print(f"\n[{idx + 1}/{total}] {status}")
    print(f"  params: {result['params']}")

    corr_strs = []
    for vname in version_names:
        v = result["version_correlations"].get(vname, float("nan"))
        corr_strs.append(f"{vname}:{v:.4f}" if not math.isnan(v) else f"{vname}:NaN")
    print(f"  version correlations: {', '.join(corr_strs)}")

    sc = result["sum_corr"]
    print(f"  sum_corr = {'NaN' if math.isnan(sc) else f'{sc:.4f}'}")

    for vname in version_names:
        k_stats = result["version_k_stats"].get(vname, {})
        k_mean = k_stats.get("k_mean", float("nan"))
        k_std = k_stats.get("k_std", float("nan"))
        n_bins = k_stats.get("nonempty_bins", float("nan"))

        k_mean_s = f"{k_mean:.2f}" if not math.isnan(k_mean) else "NaN"
        k_std_s = f"{k_std:.2f}" if not math.isnan(k_std) else "NaN"
        n_bins_s = str(int(n_bins)) if not math.isnan(n_bins) else "NaN"

        print(f"  k stats [{vname}]: mean={k_mean_s}  std={k_std_s}  nonempty_bins={n_bins_s}")

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Static correlation runner aligned with metric_definition_dynamic.py."
        )
    )

    # Version inputs
    parser.add_argument(
        "--json-dirs",
        type=str,
        required=True,
        help="Comma-separated list of per-example JSON directories, one per version",
    )
    parser.add_argument(
        "--results-jsons",
        type=str,
        required=True,
        help="Comma-separated list of evaluation_results.json paths, one per version",
    )
    parser.add_argument(
        "--version-names",
        type=str,
        default=None,
        help="Optional comma-separated version names. Defaults to v0,v1,...",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output folder",
    )
    parser.add_argument(
        "--skip-missing-results",
        action="store_true",
        help="Skip per-example JSONs not found in results JSON",
    )

    # Grid parameters aligned to metric_definition_dynamic.py
    parser.add_argument("--offsets", type=str, default="0")
    parser.add_argument("--adaptive-k-max-search-tokens", type=str, default="40")
    parser.add_argument("--adaptive-k-smooth-options", type=str, default="true")
    parser.add_argument("--adaptive-k-prior-means", type=str, default="none")
    parser.add_argument("--adaptive-k-prior-kappas", type=str, default="1.0")
    parser.add_argument("--adaptive-k-beta-concentrations", type=str, default="20.0")
    parser.add_argument("--adaptive-k-cp-prob-thresholds", type=str, default="0.20")
    parser.add_argument("--adaptive-k-pre-windows", type=str, default="5")
    parser.add_argument("--adaptive-k-post-windows", type=str, default="4")
    parser.add_argument("--adaptive-k-drop-ratios", type=str, default="0.65")
    parser.add_argument("--adaptive-k-clip-epsilons", type=str, default="1e-6")
    parser.add_argument("--ratio-bin-widths", type=str, default="0.15")

    args = parser.parse_args()

    json_dirs = [Path(p.strip()) for p in args.json_dirs.split(",") if p.strip()]
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

    offsets = parse_int_list(args.offsets)
    adaptive_k_max_search_tokens = parse_int_list(args.adaptive_k_max_search_tokens)
    adaptive_k_smooth = parse_bool_list(args.adaptive_k_smooth_options)
    adaptive_k_prior_mean = parse_optional_float_list(args.adaptive_k_prior_means)
    adaptive_k_prior_kappa = parse_float_list(args.adaptive_k_prior_kappas)
    adaptive_k_beta_concentration = parse_float_list(args.adaptive_k_beta_concentrations)
    adaptive_k_cp_prob_threshold = parse_float_list(args.adaptive_k_cp_prob_thresholds)
    adaptive_k_pre_window = parse_int_list(args.adaptive_k_pre_windows)
    adaptive_k_post_window = parse_int_list(args.adaptive_k_post_windows)
    adaptive_k_drop_ratio = parse_float_list(args.adaptive_k_drop_ratios)
    adaptive_k_clip_eps = parse_float_list(args.adaptive_k_clip_epsilons)
    ratio_bin_widths = parse_float_list(args.ratio_bin_widths)

    print("=" * 80)
    print(f"Metric: {dynamic_metric.METRIC_NAME}")
    print(f"Versions ({len(version_names)}): {version_names}")
    print(f"offsets:                           {offsets}")
    print(f"adaptive_k_max_search_tokens:     {adaptive_k_max_search_tokens}")
    print(f"adaptive_k_smooth:                {adaptive_k_smooth}")
    print(f"adaptive_k_prior_mean:            {adaptive_k_prior_mean}")
    print(f"adaptive_k_prior_kappa:           {adaptive_k_prior_kappa}")
    print(f"adaptive_k_beta_concentration:    {adaptive_k_beta_concentration}")
    print(f"adaptive_k_cp_prob_threshold:     {adaptive_k_cp_prob_threshold}")
    print(f"adaptive_k_pre_window:            {adaptive_k_pre_window}")
    print(f"adaptive_k_post_window:           {adaptive_k_post_window}")
    print(f"adaptive_k_drop_ratio:            {adaptive_k_drop_ratio}")
    print(f"adaptive_k_clip_eps:              {adaptive_k_clip_eps}")
    print(f"ratio_bin_widths:                 {ratio_bin_widths}")
    print(
        f"Success criterion: >= {SUCCESS_HIGH_COUNT}/{len(version_names)} "
        f"corr >= {SUCCESS_HIGH_CORR}, all >= {SUCCESS_MIN_CORR}, "
        f"all nonempty_bins >= {SUCCESS_MIN_NONEMPTY_BINS}"
    )
    print("=" * 80)

    versions = []
    for vname, jdir, rjson in zip(version_names, json_dirs, results_jsons):
        print(f"Loading version '{vname}' from {jdir} ...")
        records, skipped = load_records(jdir, rjson, skip_missing=args.skip_missing_results)
        print(f"  Loaded {len(records)} examples, skipped {len(skipped)} files")
        versions.append((vname, records))

    param_grid = build_param_grid(
        offsets=offsets,
        adaptive_k_max_search_tokens=adaptive_k_max_search_tokens,
        adaptive_k_smooth=adaptive_k_smooth,
        adaptive_k_prior_mean=adaptive_k_prior_mean,
        adaptive_k_prior_kappa=adaptive_k_prior_kappa,
        adaptive_k_beta_concentration=adaptive_k_beta_concentration,
        adaptive_k_cp_prob_threshold=adaptive_k_cp_prob_threshold,
        adaptive_k_pre_window=adaptive_k_pre_window,
        adaptive_k_post_window=adaptive_k_post_window,
        adaptive_k_drop_ratio=adaptive_k_drop_ratio,
        adaptive_k_clip_eps=adaptive_k_clip_eps,
        ratio_bin_widths=ratio_bin_widths,
    )

    total = len(param_grid)
    print(f"\nParameter combinations to evaluate: {total}")
    print("=" * 80)

    all_results = []
    best_successful_plot = None
    best_successful_sum_corr = -float("inf")

    for idx, params in enumerate(param_grid):
        result = evaluate_combination(versions, params)
        all_results.append(result)

        folder_name = combo_folder_name(params)
        combo_dir = output_dir / folder_name
        plot_path = save_combination(result, combo_dir, version_names[0])

        if result["successful"]:
            sc = result["sum_corr"]
            if not math.isnan(sc) and sc > best_successful_sum_corr:
                best_successful_sum_corr = sc
                best_successful_plot = plot_path

        print_combination(result, idx, total, version_names)

    print("\n" + "=" * 80)
    print("Saving top-level summary ...")
    ranked = save_top_level_summary(all_results, output_dir, best_successful_plot, version_names)

    n_success = sum(1 for r in all_results if r["successful"])
    n_fail = len(all_results) - n_success
    print(f"Total combinations: {len(all_results)}")
    print(f"  Successful:   {n_success}")
    print(f"  Unsuccessful: {n_fail}")

    if n_success > 0:
        best = ranked[0]
        print("\nBest successful combination:")
        print(f"  params:   {best['params']}")
        print(f"  sum_corr = {best['sum_corr']:.4f}")
        for vname in version_names:
            c = best["version_correlations"].get(vname, float("nan"))
            print(f"  corr [{vname}] = {c:.4f}" if not math.isnan(c) else f"  corr [{vname}] = NaN")
    else:
        print("\nNo successful combinations found.")

    print("=" * 80)


if __name__ == "__main__":
    main()