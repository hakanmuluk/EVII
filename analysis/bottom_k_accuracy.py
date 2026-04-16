#!/usr/bin/env python3
"""
Bottom-K Accuracy Analysis

For each version (model × dataset combination), computes the metric defined in
metric_definition_dynamic.py for every example, then reports:

  - Overall accuracy across all valid examples
  - Accuracy among the bottom 10%, 20%, and 30% lowest-scoring examples

Results are printed as a table and saved to <output-dir>/bottom_k_accuracy.{json,csv}.

Usage:
    python bottom_k_accuracy.py \\
        --json-dirs               d1,d2,...,d6 \\
        --results-jsons           r1.json,...,r6.json \\
        --other-metrics-json-dirs o1,...,o6 \\
        --version-names           8b_erqa,8b_robo2vlm,8b_spatial_mm,30b_erqa,30b_robo2vlm,30b_spatial_mm \\
        --output-dir              bottom_k_accuracy_run \\
        --offset                  0 \\
        --adaptive-k-cp-prob-threshold  0.15 \\
        --adaptive-k-drop-ratio         0.65 \\
        --adaptive-k-beta-concentration 20.0 \\
        --skip-missing-results
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import metric_definition_dynamic as dynamic_metric

# ---------------------------------------------------------------------------
# Data model (mirrors analyze_auroc_auprc.py)
# ---------------------------------------------------------------------------

@dataclass
class ExampleRecord:
    example_id: str
    source_path: Path
    correct: int
    noimage_js: Optional[np.ndarray]
    total_tokens: int
    avg_log_prob: float
    avg_neg_entropy: float
    avg_neg_perplexity: float
    avg_self_certainty: float


# ---------------------------------------------------------------------------
# Data loading helpers (adapted from analyze_auroc_auprc.py)
# ---------------------------------------------------------------------------

def _extract_noimage_js(data: dict) -> np.ndarray:
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
                if items and isinstance(items[0], dict):
                    if "token_index" in items[0]:
                        items = sorted(items, key=lambda z: z["token_index"])
                    for item in items:
                        if "js_divergence" in item:
                            js_vals.append(float(item["js_divergence"]))
                        elif "js" in item:
                            js_vals.append(float(item["js"]))
                        else:
                            raise KeyError(
                                f"Could not find js field in noimage item under '{key}'"
                            )
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
                    raise KeyError(
                        "Could not find js field in noimage_comparison_trace list item"
                    )
            return np.asarray(js_vals, dtype=float)

    raise KeyError("Unsupported 'noimage_comparison_trace' format")


def load_results(
    results_json_path: Path,
) -> Tuple[Dict[str, int], Dict[str, int]]:
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

    noimage_js: Optional[np.ndarray] = None
    total_tokens: int = 0
    if "js_trace" in data and "noimage_comparison_trace" in data:
        noimage_js = _extract_noimage_js(data)
        js_matrix = np.asarray(data["js_trace"]["js_matrix"], dtype=float)
        _, total_tokens = js_matrix.shape
    elif "js_trace" in data:
        js_matrix = np.asarray(data["js_trace"]["js_matrix"], dtype=float)
        _, total_tokens = js_matrix.shape

    nan = float("nan")
    avg_log_prob       = nan
    avg_neg_entropy    = nan
    avg_neg_perplexity = nan
    avg_self_certainty = nan

    if "token_metrics_trace" in data:
        tmt = data["token_metrics_trace"]
        log_probs        = np.asarray(tmt["log_probs"],        dtype=float)
        entropies        = np.asarray(tmt["entropies"],        dtype=float)
        self_certainties = np.asarray(tmt["self_certainties"], dtype=float)
        avg_log_prob       = float(np.mean(log_probs))
        avg_neg_entropy    = float(-np.mean(entropies))
        avg_neg_perplexity = float(-np.mean(np.exp(entropies)))
        avg_self_certainty = float(np.mean(self_certainties))

    return ExampleRecord(
        example_id=example_id,
        source_path=json_path,
        correct=int(correct),
        noimage_js=noimage_js,
        total_tokens=int(total_tokens),
        avg_log_prob=avg_log_prob,
        avg_neg_entropy=avg_neg_entropy,
        avg_neg_perplexity=avg_neg_perplexity,
        avg_self_certainty=avg_self_certainty,
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

            if "example" not in data:
                skipped.append((str(json_path), "missing 'example' key"))
                continue
            if "js_trace" not in data and "token_metrics_trace" not in data:
                skipped.append((str(json_path), "missing both 'js_trace' and 'token_metrics_trace'"))
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

            record = load_example_record(json_path=json_path, correct=int(correct))
            records.append(record)

        except Exception as exc:
            skipped.append((str(json_path), str(exc)))

    if not records:
        raise RuntimeError(
            f"No usable per-example JSON files loaded from {json_dir}"
        )

    return records, skipped


def load_other_metrics_dir(dir_path: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}

    for json_path in sorted(dir_path.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "example" not in data or "token_metrics_trace" not in data:
                continue

            example_id = data["example"]["example_id"]
            tmt = data["token_metrics_trace"]

            log_probs        = np.asarray(tmt["log_probs"],        dtype=float)
            entropies        = np.asarray(tmt["entropies"],        dtype=float)
            self_certainties = np.asarray(tmt["self_certainties"], dtype=float)

            result[example_id] = {
                "avg_log_prob":       float(np.mean(log_probs)),
                "avg_neg_entropy":    float(-np.mean(entropies)),
                "avg_neg_perplexity": float(-np.mean(np.exp(entropies))),
                "avg_self_certainty": float(np.mean(self_certainties)),
            }

        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_bottom_k_accuracy(
    records: List[ExampleRecord],
    bocpd_params: dict,
    percentiles: List[float] = (0.10, 0.20, 0.30),
) -> dict:
    """
    Compute metric for each record, sort ascending, then compute accuracy
    for the overall set and each bottom-k percentile slice.

    Returns a dict with:
        n_total      – number of loaded records
        n_valid      – records with a finite metric value
        overall_acc  – accuracy over all valid records
        bottom_k     – dict keyed by percentile string (e.g. "0.10") with
                       keys: n, correct, accuracy
        per_example  – list of {example_id, metric_value, correct} sorted asc
    """
    scored: List[Tuple[float, int, str]] = []  # (metric_value, correct, example_id)
    n_total = len(records)

    for rec in records:
        if rec.noimage_js is None:
            continue
        try:
            result = dynamic_metric.compute_metric_for_example(
                record=rec, **bocpd_params
            )
            val = float(result["metric_value"])
        except Exception:
            continue

        if not math.isfinite(val):
            continue

        scored.append((val, int(rec.correct), rec.example_id))

    n_valid = len(scored)

    if n_valid == 0:
        nan = float("nan")
        bottom_k_results = {
            f"{p:.2f}": {"n": 0, "correct": 0, "accuracy": nan}
            for p in percentiles
        }
        return {
            "n_total": n_total,
            "n_valid": 0,
            "overall_acc": nan,
            "bottom_k": bottom_k_results,
            "per_example": [],
        }

    # Sort by metric value ascending (lowest scores first)
    scored.sort(key=lambda x: x[0], reverse=False)

    overall_correct = sum(c for _, c, _ in scored)
    overall_acc = overall_correct / n_valid

    bottom_k_results: dict = {}
    for p in percentiles:
        k = max(1, math.ceil(n_valid * p))
        subset = scored[:k]
        n_correct = sum(c for _, c, _ in subset)
        bottom_k_results[f"{p:.2f}"] = {
            "n": k,
            "correct": n_correct,
            "accuracy": n_correct / k,
        }

    per_example = [
        {"example_id": eid, "metric_value": val, "correct": corr}
        for val, corr, eid in scored
    ]

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "overall_correct": overall_correct,
        "overall_acc": overall_acc,
        "bottom_k": bottom_k_results,
        "per_example": per_example,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_acc(acc) -> str:
    if math.isnan(acc):
        return "  N/A  "
    return f"{acc * 100:6.2f}%"


def print_results_table(
    version_results: List[Tuple[str, dict]],
    percentiles: List[float],
) -> None:
    pct_headers = [f"bot{int(p * 100)}%" for p in percentiles]
    pct_keys    = [f"{p:.2f}" for p in percentiles]

    col_w = 14
    header_parts = (
        ["version".ljust(24)]
        + ["N_total".rjust(8), "N_valid".rjust(8)]
        + ["overall_acc".rjust(col_w)]
        + [h.rjust(col_w) for h in pct_headers]
        + [f"n({h})".rjust(8) for h in pct_headers]
    )
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for vname, res in version_results:
        row_parts = (
            [vname.ljust(24)]
            + [str(res["n_total"]).rjust(8), str(res["n_valid"]).rjust(8)]
            + [_fmt_acc(res["overall_acc"]).rjust(col_w)]
            + [_fmt_acc(res["bottom_k"][k]["accuracy"]).rjust(col_w) for k in pct_keys]
            + [str(res["bottom_k"][k]["n"]).rjust(8) for k in pct_keys]
        )
        print("  ".join(row_parts))

    print(sep)
    print()


def save_json(
    version_results: List[Tuple[str, dict]],
    params: dict,
    output_path: Path,
) -> None:
    out = {
        "metric_name": dynamic_metric.METRIC_NAME,
        "bocpd_params": params,
        "versions": {vname: res for vname, res in version_results},
    }

    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_default)


def save_csv(
    version_results: List[Tuple[str, dict]],
    percentiles: List[float],
    output_path: Path,
) -> None:
    pct_keys = [f"{p:.2f}" for p in percentiles]

    fieldnames = (
        ["version", "n_total", "n_valid", "overall_correct", "overall_acc"]
        + [f"bot{int(float(k)*100)}_n" for k in pct_keys]
        + [f"bot{int(float(k)*100)}_correct" for k in pct_keys]
        + [f"bot{int(float(k)*100)}_acc" for k in pct_keys]
    )

    rows = []
    for vname, res in version_results:
        row: dict = {
            "version":         vname,
            "n_total":         res["n_total"],
            "n_valid":         res["n_valid"],
            "overall_correct": res.get("overall_correct", ""),
            "overall_acc":     res["overall_acc"] if math.isfinite(res["overall_acc"]) else "",
        }
        for k in pct_keys:
            pct_int = int(float(k) * 100)
            bk = res["bottom_k"][k]
            row[f"bot{pct_int}_n"]       = bk["n"]
            row[f"bot{pct_int}_correct"]  = bk["correct"]
            acc = bk["accuracy"]
            row[f"bot{pct_int}_acc"] = acc if math.isfinite(acc) else ""
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bottom-K accuracy analysis: for each version, rank examples by the "
            "dynamic BOCPD metric and report accuracy at the bottom 10/20/30%."
        )
    )

    # Multi-version inputs
    parser.add_argument(
        "--json-dirs", type=str, required=True,
        help="Comma-separated directories with js_trace / noimage_comparison_trace, one per version",
    )
    parser.add_argument(
        "--results-jsons", type=str, required=True,
        help="Comma-separated evaluation_results.json paths, one per version",
    )
    parser.add_argument(
        "--other-metrics-json-dirs", type=str, default=None,
        help=(
            "Optional comma-separated directories with token_metrics_trace, one per version. "
            "Use an empty string for a version that has no separate metrics dir."
        ),
    )
    parser.add_argument(
        "--version-names", type=str, default=None,
        help="Optional comma-separated version names (default: v0,v1,...)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output folder for bottom_k_accuracy.json and bottom_k_accuracy.csv",
    )
    parser.add_argument(
        "--skip-missing-results", action="store_true",
        help="Skip per-example JSONs not found in results JSON",
    )
    parser.add_argument(
        "--percentiles", type=str, default="0.10,0.20,0.30",
        help="Comma-separated bottom-k percentiles to evaluate (default: 0.10,0.20,0.30)",
    )

    # BOCPD hyperparameters (single values, not lists)
    parser.add_argument("--offset",                        type=int,   default=0)
    parser.add_argument("--adaptive-k-max-search-tokens",  type=int,   default=40)
    parser.add_argument("--adaptive-k-smooth",             type=str,   default="true")
    parser.add_argument("--adaptive-k-prior-mean",         type=str,   default="none")
    parser.add_argument("--adaptive-k-prior-kappa",        type=float, default=1.0)
    parser.add_argument("--adaptive-k-beta-concentration", type=float, default=20.0)
    parser.add_argument("--adaptive-k-cp-prob-threshold",  type=float, default=0.1)
    parser.add_argument("--adaptive-k-pre-window",         type=int,   default=5)
    parser.add_argument("--adaptive-k-post-window",        type=int,   default=4)
    parser.add_argument("--adaptive-k-drop-ratio",         type=float, default=0.7)
    parser.add_argument("--adaptive-k-clip-eps",           type=float, default=1e-6)

    args = parser.parse_args()

    # --- Parse multi-version inputs ---
    json_dirs     = [Path(p.strip()) for p in args.json_dirs.split(",")     if p.strip()]
    results_jsons = [Path(p.strip()) for p in args.results_jsons.split(",") if p.strip()]

    if len(json_dirs) != len(results_jsons):
        parser.error(
            f"--json-dirs ({len(json_dirs)}) and --results-jsons "
            f"({len(results_jsons)}) must have the same number of entries."
        )

    if args.other_metrics_json_dirs:
        raw_other = args.other_metrics_json_dirs.split(",")
        other_dirs: List[Optional[Path]] = [
            Path(p.strip()) if p.strip() else None for p in raw_other
        ]
        if len(other_dirs) != len(json_dirs):
            parser.error(
                f"--other-metrics-json-dirs ({len(other_dirs)}) must match "
                f"the number of versions ({len(json_dirs)})."
            )
    else:
        other_dirs = [None] * len(json_dirs)

    if args.version_names:
        version_names = [n.strip() for n in args.version_names.split(",") if n.strip()]
        if len(version_names) != len(json_dirs):
            parser.error(
                f"--version-names ({len(version_names)}) must match "
                f"the number of versions ({len(json_dirs)})."
            )
    else:
        version_names = [f"v{i}" for i in range(len(json_dirs))]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse percentiles ---
    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]

    # --- Parse BOCPD params ---
    smooth_str = args.adaptive_k_smooth.strip().lower()
    smooth = smooth_str in {"1", "true", "t", "yes", "y"}

    prior_mean_str = args.adaptive_k_prior_mean.strip().lower()
    prior_mean: Optional[float] = None if prior_mean_str in {"none", "null"} else float(prior_mean_str)

    bocpd_params = {
        "offset":                        args.offset,
        "adaptive_k_max_search_tokens":  args.adaptive_k_max_search_tokens,
        "adaptive_k_smooth":             smooth,
        "adaptive_k_prior_mean":         prior_mean,
        "adaptive_k_prior_kappa":        args.adaptive_k_prior_kappa,
        "adaptive_k_beta_concentration": args.adaptive_k_beta_concentration,
        "adaptive_k_cp_prob_threshold":  args.adaptive_k_cp_prob_threshold,
        "adaptive_k_pre_window":         args.adaptive_k_pre_window,
        "adaptive_k_post_window":        args.adaptive_k_post_window,
        "adaptive_k_drop_ratio":         args.adaptive_k_drop_ratio,
        "adaptive_k_clip_eps":           args.adaptive_k_clip_eps,
    }

    print("=" * 80)
    print(f"Bottom-K Accuracy Analysis  —  {dynamic_metric.METRIC_NAME}")
    print(f"Versions ({len(version_names)}): {version_names}")
    print(f"Percentiles: {percentiles}")
    print(f"BOCPD params: {bocpd_params}")
    print("=" * 80)

    # --- Load all versions and run analysis ---
    version_results: List[Tuple[str, dict]] = []

    for vname, jdir, rjson, odir in zip(version_names, json_dirs, results_jsons, other_dirs):
        print(f"\nLoading version '{vname}' from {jdir} …")
        records, skipped = load_records(jdir, rjson, skip_missing=args.skip_missing_results)
        print(f"  Loaded {len(records)} examples, skipped {len(skipped)} files")
        if skipped:
            for path, reason in skipped[:3]:
                print(f"    skip: {Path(path).name} — {reason}")
            if len(skipped) > 3:
                print(f"    … and {len(skipped) - 3} more")

        if odir is not None:
            print(f"  Merging other-metrics from {odir} …")
            other_metrics = load_other_metrics_dir(odir)
            merged = 0
            for rec in records:
                if rec.example_id in other_metrics:
                    m = other_metrics[rec.example_id]
                    rec.avg_log_prob       = m["avg_log_prob"]
                    rec.avg_neg_entropy    = m["avg_neg_entropy"]
                    rec.avg_neg_perplexity = m["avg_neg_perplexity"]
                    rec.avg_self_certainty = m["avg_self_certainty"]
                    merged += 1
            print(f"  Merged other-metrics for {merged}/{len(records)} examples")

        print(f"  Computing metric …")
        res = compute_bottom_k_accuracy(records, bocpd_params, percentiles=percentiles)
        version_results.append((vname, res))

        pct_keys = [f"{p:.2f}" for p in percentiles]
        print(
            f"  n_valid={res['n_valid']}  overall_acc={_fmt_acc(res['overall_acc'])}  "
            + "  ".join(
                f"bot{int(float(k)*100)}%={_fmt_acc(res['bottom_k'][k]['accuracy'])} (n={res['bottom_k'][k]['n']})"
                for k in pct_keys
            )
        )

    # --- Print summary table ---
    print_results_table(version_results, percentiles)

    # --- Save outputs ---
    json_path = output_dir / "bottom_k_accuracy.json"
    csv_path  = output_dir / "bottom_k_accuracy.csv"

    save_json(version_results, bocpd_params, json_path)
    save_csv(version_results, percentiles, csv_path)

    print(f"Results saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
