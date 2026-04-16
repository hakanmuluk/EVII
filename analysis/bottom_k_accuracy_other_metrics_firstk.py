#!/usr/bin/env python3
"""
Bottom-K accuracy analysis for simple token-level metrics, computed over the
**first k generated tokens** instead of all tokens.

k is determined per-example by one of three strategies:
  bocpd       — Beta-family BOCPD on the no_image_js signal inside
                token_metrics_trace (same algorithm as metric_definition_dynamic.py)
  constant:N  — exactly N tokens (capped to total_tokens)
  ratio:R     — round(R * total_tokens), clamped to [1, total_tokens]

Metrics evaluated (over first-k tokens):
  avg_log_prob         lower = less confident
  avg_neg_entropy      lower = higher entropy = less confident
  avg_neg_perplexity   lower = higher perplexity = less confident
  avg_self_certainty   lower = less certain

For each (k_spec, metric, version) combination the script ranks examples by the
first-k metric value (ascending) and reports accuracy at the overall level and
at the bottom 10/20/30% thresholds.

Results are printed as tables and saved to:
  <output-dir>/bottom_k_accuracy_firstk.json
  <output-dir>/bottom_k_accuracy_firstk.csv

Usage:
    python bottom_k_accuracy_other_metrics_firstk.py \\
        --k-specs "bocpd,constant:10,constant:30,constant:50,ratio:0.05,ratio:0.10,ratio:0.15" \\
        --percentiles "0.10,0.20,0.30" \\
        --output-dir bottom_k_firstk_run
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded 6 cases — token_metrics_trace lives in the other_metrics dirs
# ---------------------------------------------------------------------------

CASES: List[Tuple[str, str, str]] = [
    ("8b_erqa",     "erqa_8b_other_metrics",        "qwen3vl_8b_evaluation_results_erqa.json"),
    ("8b_robo",     "robo2vlm_8b_other_metrics",    "qwen3vl_8b_evaluation_results_robo2vlm.json"),
    ("8b_spatial",  "spatial_mm_8b_other_metrics",  "qwen3vl_8b_evaluation_results_spatial_mm.json"),
    ("30b_erqa",    "erqa_30b_other_metrics",        "qwen3vl_30b_evaluation_results_erqa.json"),
    ("30b_robo",    "robo2vlm_30b_other_metrics",    "qwen3vl_30b_evaluation_results_robo2vlm.json"),
    ("30b_spatial", "spatial_mm_30b_other_metrics",  "qwen3vl_30b_evaluation_results_spatial_mm.json"),
]

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRIC_NAMES: List[str] = [
    "avg_log_prob",
    "avg_neg_entropy",
    "avg_neg_perplexity",
    "avg_self_certainty",
]

METRIC_LABELS: Dict[str, str] = {
    "avg_log_prob":        "Avg log-probability (first k)",
    "avg_neg_entropy":     "Avg negative entropy, nats (first k)",
    "avg_neg_perplexity":  "Avg negative perplexity, exp H (first k)",
    "avg_self_certainty":  "Avg self-certainty (first k)",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExampleRecord:
    example_id: str
    source_path: Path
    correct: int
    log_probs: np.ndarray
    entropies: np.ndarray
    self_certainties: np.ndarray
    noimage_js: np.ndarray          # from token_metrics_trace["no_image_js"]
    total_tokens: int

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results_json(
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


def load_records(
    other_metrics_dir: Path,
    results_json: Path,
    skip_missing: bool,
) -> Tuple[List[ExampleRecord], List[Tuple[str, str]]]:
    by_example_id, by_filename_stem = _load_results_json(results_json)

    records: List[ExampleRecord] = []
    skipped: List[Tuple[str, str]] = []

    for json_path in sorted(other_metrics_dir.glob("*.json")):
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

            tmt = data["token_metrics_trace"]
            log_probs        = np.asarray(tmt["log_probs"],        dtype=float)
            entropies        = np.asarray(tmt["entropies"],         dtype=float)
            self_certainties = np.asarray(tmt["self_certainties"],  dtype=float)
            noimage_js_raw   = tmt.get("no_image_js")
            noimage_js       = (
                np.asarray(noimage_js_raw, dtype=float)
                if noimage_js_raw is not None
                else np.asarray([], dtype=float)
            )

            records.append(ExampleRecord(
                example_id=example_id,
                source_path=json_path,
                correct=int(correct),
                log_probs=log_probs,
                entropies=entropies,
                self_certainties=self_certainties,
                noimage_js=noimage_js,
                total_tokens=len(log_probs),
            ))

        except Exception as e:
            skipped.append((str(json_path), str(e)))

    if not records:
        raise RuntimeError(
            f"No usable per-example JSON files loaded from {other_metrics_dir}"
        )

    return records, skipped

# ---------------------------------------------------------------------------
# BOCPD helpers  (ported from metric_definition_dynamic.py)
# ---------------------------------------------------------------------------

def _smooth_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    return np.convolve(x, np.array([0.25, 0.5, 0.25]), mode="same")


def _clip_unit(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(float(x), eps, 1.0 - eps))


def _beta_pdf(x: float, alpha: float, beta_p: float, eps: float = 1e-12) -> float:
    x      = _clip_unit(x, max(eps, 1e-12))
    alpha  = max(float(alpha), eps)
    beta_p = max(float(beta_p), eps)
    log_norm = math.lgamma(alpha) + math.lgamma(beta_p) - math.lgamma(alpha + beta_p)
    log_pdf  = (alpha - 1.0) * math.log(x) + (beta_p - 1.0) * math.log1p(-x) - log_norm
    return 0.0 if log_pdf < -745.0 else math.exp(log_pdf)


def _bocpd_cp_probs(
    x: np.ndarray,
    hazard: float,
    prior_mean: float,
    prior_kappa: float,
    beta_concentration: float,
    clip_eps: float,
) -> np.ndarray:
    T = len(x)
    if T == 0:
        return np.zeros(0, dtype=float)

    x = np.clip(x, clip_eps, 1.0 - clip_eps)
    R = np.zeros((T + 1, T + 1), dtype=float)
    R[0, 0] = 1.0

    count_p = np.array([0.0])
    sum_p   = np.array([0.0])
    cp_probs = []

    for t in range(1, T + 1):
        xt = float(x[t - 1])
        prev = t
        pred = np.zeros(prev, dtype=float)

        for r in range(prev):
            post_mean = (prior_kappa * prior_mean + sum_p[r]) / (prior_kappa + count_p[r])
            post_mean = _clip_unit(post_mean, clip_eps)
            alpha_r   = max(post_mean * beta_concentration, clip_eps)
            beta_r    = max((1.0 - post_mean) * beta_concentration, clip_eps)
            pred[r]   = _beta_pdf(xt, alpha_r, beta_r, clip_eps)

        growth = R[t - 1, :prev] * pred * (1.0 - hazard)
        cp     = float(np.sum(R[t - 1, :prev] * pred * hazard))

        joint = np.zeros(t + 1, dtype=float)
        joint[0]  = cp
        joint[1:] = growth

        ev = float(np.sum(joint))
        if ev <= 1e-300:
            joint[:] = 1.0 / len(joint)
            ev = 1.0

        R[t, : t + 1] = joint / ev
        cp_probs.append(float(R[t, 0]))

        new_c = np.empty(t + 1)
        new_s = np.empty(t + 1)
        new_c[0] = new_s[0] = 0.0
        for r in range(1, t + 1):
            new_c[r] = count_p[r - 1] + 1.0
            new_s[r] = sum_p[r - 1] + xt
        count_p = new_c
        sum_p   = new_s

    return np.asarray(cp_probs, dtype=float)


def _choose_k_bocpd(
    noimage_js: np.ndarray,
    total_tokens: int,
    offset: int,
    max_search_tokens: int,
    smooth: bool,
    prior_kappa: float,
    beta_concentration: float,
    cp_prob_threshold: float,
    pre_window: int,
    post_window: int,
    drop_ratio: float,
    clip_eps: float = 1e-6,
) -> int:
    total_tokens = int(total_tokens)
    if total_tokens <= 0 or len(noimage_js) == 0:
        return 0

    # Align noimage_js with the token dimension
    aligned = np.full(total_tokens, np.nan, dtype=float)
    for i in range(total_tokens):
        j = i + offset
        if 0 <= j < len(noimage_js):
            aligned[i] = noimage_js[j]

    k_min              = int(total_tokens * 0.05)
    expected_run_len   = max(int(total_tokens * 0.1), 1)
    k_max              = int(total_tokens * 0.15)
    search_end         = min(total_tokens, max_search_tokens, k_max)

    if search_end <= 0:
        return 0

    x = np.nan_to_num(aligned[:search_end], nan=0.0)
    x = np.clip(x, clip_eps, 1.0 - clip_eps)
    if smooth:
        x = _smooth_1d(x)
        x = np.clip(x, clip_eps, 1.0 - clip_eps)

    prior_mean = _clip_unit(float(np.mean(x)), clip_eps)
    hazard     = 1.0 / float(expected_run_len)

    cp = _bocpd_cp_probs(x, hazard, prior_mean, prior_kappa,
                         beta_concentration, clip_eps)

    lower = max(1, k_min)
    upper = min(search_end, k_max)

    if lower > upper:
        return max(0, upper)

    for t in range(lower, upper + 1):
        cp_t      = float(cp[t - 1])
        pre_s     = max(0, t - pre_window)
        pre_mean  = float(np.mean(x[pre_s:t]))  if t > pre_s else 0.0
        post_e    = min(search_end, t + post_window)
        post_mean = float(np.mean(x[t:post_e])) if post_e > t  else 0.0
        downward  = post_mean <= drop_ratio * (pre_mean + 1e-12)
        if cp_t >= cp_prob_threshold and downward:
            return t

    window   = cp[lower - 1 : upper]
    best_idx = int(np.argmax(window))
    return lower + best_idx

# ---------------------------------------------------------------------------
# k resolution
# ---------------------------------------------------------------------------

def resolve_k(
    k_spec: str,
    record: ExampleRecord,
    offset: int,
    bocpd_params: dict,
) -> int:
    kind, _, raw = k_spec.partition(":")
    kind = kind.strip().lower()
    total = record.total_tokens

    if total <= 0:
        return 0

    if kind == "constant":
        return min(int(raw.strip()), total)

    if kind == "ratio":
        return max(1, min(int(round(float(raw.strip()) * total)), total))

    if kind == "bocpd":
        return _choose_k_bocpd(
            noimage_js=record.noimage_js,
            total_tokens=total,
            offset=offset,
            max_search_tokens=bocpd_params["max_search_tokens"],
            smooth=bocpd_params["smooth"],
            prior_kappa=bocpd_params["prior_kappa"],
            beta_concentration=bocpd_params["beta_concentration"],
            cp_prob_threshold=bocpd_params["cp_prob_threshold"],
            pre_window=bocpd_params["pre_window"],
            post_window=bocpd_params["post_window"],
            drop_ratio=bocpd_params["drop_ratio"],
            clip_eps=bocpd_params["clip_eps"],
        )

    raise ValueError(f"Unknown k_spec: {k_spec!r}")

# ---------------------------------------------------------------------------
# Per-example first-k metric computation
# ---------------------------------------------------------------------------

def compute_firstk_metrics(
    record: ExampleRecord,
    k_spec: str,
    offset: int,
    bocpd_params: dict,
) -> Dict[str, float]:
    """Return dict of {metric_name: float} for the first-k token window."""
    k = resolve_k(k_spec, record, offset, bocpd_params)

    if k <= 0:
        nan = float("nan")
        return {m: nan for m in METRIC_NAMES}

    lp  = record.log_probs[:k]
    ent = record.entropies[:k]
    sc  = record.self_certainties[:k]

    return {
        "avg_log_prob":       float(np.mean(lp)),
        "avg_neg_entropy":    float(-np.mean(ent)),
        "avg_neg_perplexity": float(-np.mean(np.exp(ent))),
        "avg_self_certainty": float(np.mean(sc)),
    }

# ---------------------------------------------------------------------------
# Bottom-k accuracy computation
# ---------------------------------------------------------------------------

def compute_bottom_k_accuracy(
    scored: List[Tuple[float, int, str]],
    percentiles: List[float],
) -> dict:
    """
    Given a list of (metric_value, correct, example_id) already filtered for
    finite values, sort ascending and compute bottom-k accuracy at each percentile.
    """
    n_valid = len(scored)

    if n_valid == 0:
        nan = float("nan")
        return {
            "n_valid":    0,
            "overall_acc": nan,
            "bottom_k": {
                f"{p:.2f}": {"n": 0, "correct": 0, "accuracy": nan}
                for p in percentiles
            },
        }

    scored = sorted(scored, key=lambda x: x[0])
    overall_correct = sum(c for _, c, _ in scored)
    overall_acc     = overall_correct / n_valid

    bottom_k: dict = {}
    for p in percentiles:
        k  = max(1, math.ceil(n_valid * p))
        sub = scored[:k]
        nc = sum(c for _, c, _ in sub)
        bottom_k[f"{p:.2f}"] = {"n": k, "correct": nc, "accuracy": nc / k}

    return {
        "n_valid":     n_valid,
        "overall_correct": overall_correct,
        "overall_acc": overall_acc,
        "bottom_k":    bottom_k,
    }

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _fmt(acc) -> str:
    return "  N/A  " if (not math.isfinite(acc)) else f"{acc * 100:6.2f}%"


def _fmt_delta(delta: float) -> str:
    """Format a percentage-point delta with sign, e.g. '+3.21pp'."""
    if not math.isfinite(delta):
        return "  N/A  "
    return f"{delta * 100:+6.2f}pp"


def print_table(
    k_spec: str,
    metric_name: str,
    version_results: List[Tuple[str, dict]],
    percentiles: List[float],
) -> None:
    pct_hdrs = [f"bot{int(p * 100)}%" for p in percentiles]
    pct_keys = [f"{p:.2f}" for p in percentiles]
    col_w = 14

    header_parts = (
        ["version".ljust(24)]
        + ["N_valid".rjust(8)]
        + ["overall".rjust(col_w)]
        + [h.rjust(col_w) for h in pct_hdrs]
        + [f"Δ({h})".rjust(col_w) for h in pct_hdrs]
        + [f"n({h})".rjust(8) for h in pct_hdrs]
    )
    header = "  ".join(header_parts)
    sep    = "-" * len(header)

    label = METRIC_LABELS.get(metric_name, metric_name)
    print(f"\n{'=' * len(header)}")
    print(f"k={k_spec}  |  Metric: {label}")
    print(sep)
    print(header)
    print(sep)

    # Collect deltas per percentile to compute the average across cases
    deltas_per_pct: Dict[str, List[float]] = {k: [] for k in pct_keys}

    for vname, res in version_results:
        overall = res["overall_acc"]
        deltas = []
        for k in pct_keys:
            bot_acc = res["bottom_k"][k]["accuracy"]
            if math.isfinite(overall) and math.isfinite(bot_acc):
                d = bot_acc - overall
                deltas_per_pct[k].append(d)
                deltas.append(d)
            else:
                deltas.append(float("nan"))

        row_parts = (
            [vname.ljust(24)]
            + [str(res["n_valid"]).rjust(8)]
            + [_fmt(overall).rjust(col_w)]
            + [_fmt(res["bottom_k"][k]["accuracy"]).rjust(col_w) for k in pct_keys]
            + [_fmt_delta(d).rjust(col_w) for d in deltas]
            + [str(res["bottom_k"][k]["n"]).rjust(8) for k in pct_keys]
        )
        print("  ".join(row_parts))

    # Average delta row
    print(sep)
    avg_deltas = [
        float(np.mean(deltas_per_pct[k])) if deltas_per_pct[k] else float("nan")
        for k in pct_keys
    ]
    avg_row_parts = (
        ["avg Δ (all cases)".ljust(24)]
        + ["".rjust(8)]
        + ["".rjust(col_w)]
        + ["".rjust(col_w) for _ in pct_keys]
        + [_fmt_delta(d).rjust(col_w) for d in avg_deltas]
        + ["".rjust(8) for _ in pct_keys]
    )
    print("  ".join(avg_row_parts))
    print(sep)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, Path):        return str(obj)
    return str(obj)


def save_json(results: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)


def save_csv(
    results: dict,
    percentiles: List[float],
    case_names: List[str],
    path: Path,
) -> None:
    pct_keys = [f"{p:.2f}" for p in percentiles]
    fieldnames = (
        ["k_spec", "metric", "version", "n_valid", "overall_correct", "overall_acc"]
        + [f"bot{int(float(k)*100)}_n"       for k in pct_keys]
        + [f"bot{int(float(k)*100)}_correct"  for k in pct_keys]
        + [f"bot{int(float(k)*100)}_acc"      for k in pct_keys]
    )

    rows = []
    for k_spec, per_metric in results.items():
        for metric_name, per_version in per_metric.items():
            for vname in case_names:
                res = per_version.get(vname, {})
                pct_int_keys = [int(float(k) * 100) for k in pct_keys]
                row: dict = {
                    "k_spec":          k_spec,
                    "metric":          metric_name,
                    "version":         vname,
                    "n_valid":         res.get("n_valid", ""),
                    "overall_correct": res.get("overall_correct", ""),
                    "overall_acc":     (
                        res["overall_acc"]
                        if res and math.isfinite(res.get("overall_acc", float("nan")))
                        else ""
                    ),
                }
                for k, pi in zip(pct_keys, pct_int_keys):
                    bk = res.get("bottom_k", {}).get(k, {})
                    row[f"bot{pi}_n"]       = bk.get("n", "")
                    row[f"bot{pi}_correct"]  = bk.get("correct", "")
                    acc = bk.get("accuracy", float("nan"))
                    row[f"bot{pi}_acc"]     = acc if (bk and math.isfinite(acc)) else ""
                rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bottom-K accuracy analysis for token-level metrics computed over "
            "the FIRST k generated tokens. k is set per-example via BOCPD, "
            "a constant value, or a ratio of total tokens."
        )
    )

    parser.add_argument(
        "--k-specs", type=str,
        default="bocpd,constant:10,constant:30,constant:50,ratio:0.05,ratio:0.10,ratio:0.15",
        help="Comma-separated k specs: 'bocpd', 'constant:N', 'ratio:R'.",
    )
    parser.add_argument(
        "--bocpd-offset", type=int, default=0,
        help="Vision offset used when aligning noimage_js for BOCPD k-selection (default: 0).",
    )
    parser.add_argument(
        "--percentiles", type=str, default="0.10,0.20,0.30",
        help="Comma-separated bottom-k percentiles (default: 0.10,0.20,0.30).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="bottom_k_firstk_run",
        help="Output folder (default: bottom_k_firstk_run).",
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base directory containing the 6 case folders. Defaults to script parent.",
    )
    parser.add_argument(
        "--skip-missing-results", action="store_true",
        help="Skip per-example JSONs not found in the results JSON.",
    )
    parser.add_argument(
        "--metrics", type=str, default=",".join(METRIC_NAMES),
        help=f"Comma-separated subset of metrics (default: all four).",
    )

    # BOCPD hyperparameters
    parser.add_argument("--bocpd-max-search-tokens",  type=int,   default=40)
    parser.add_argument("--bocpd-no-smooth",          action="store_true")
    parser.add_argument("--bocpd-prior-kappa",        type=float, default=1.0)
    parser.add_argument("--bocpd-beta-concentration", type=float, default=20.0)
    parser.add_argument("--bocpd-cp-prob-threshold",  type=float, default=0.1)
    parser.add_argument("--bocpd-pre-window",         type=int,   default=5)
    parser.add_argument("--bocpd-post-window",        type=int,   default=4)
    parser.add_argument("--bocpd-drop-ratio",         type=float, default=0.7)

    args = parser.parse_args()

    base_dir   = Path(args.base_dir) if args.base_dir else Path(__file__).parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k_specs    = [s.strip() for s in args.k_specs.split(",") if s.strip()]
    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    metrics_to_run = [m.strip() for m in args.metrics.split(",") if m.strip()]

    unknown = [m for m in metrics_to_run if m not in METRIC_NAMES]
    if unknown:
        parser.error(f"Unknown metrics: {unknown}. Valid: {METRIC_NAMES}")

    bocpd_params = {
        "max_search_tokens": args.bocpd_max_search_tokens,
        "smooth":            not args.bocpd_no_smooth,
        "prior_kappa":       args.bocpd_prior_kappa,
        "beta_concentration":args.bocpd_beta_concentration,
        "cp_prob_threshold": args.bocpd_cp_prob_threshold,
        "pre_window":        args.bocpd_pre_window,
        "post_window":       args.bocpd_post_window,
        "drop_ratio":        args.bocpd_drop_ratio,
        "clip_eps":          1e-6,
    }

    print("=" * 80)
    print("Bottom-K Accuracy — first-k token metrics")
    print(f"k_specs:     {k_specs}")
    print(f"bocpd offset:{args.bocpd_offset}  (only for bocpd k-selection)")
    print(f"percentiles: {percentiles}")
    print(f"metrics:     {metrics_to_run}")
    print(f"output_dir:  {output_dir}")
    print(f"base_dir:    {base_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load all 6 cases
    # ------------------------------------------------------------------
    case_records: List[Tuple[str, List[ExampleRecord]]] = []
    case_names:   List[str] = []

    for cname, other_metrics_rel, results_json_rel in CASES:
        other_metrics_dir = base_dir / other_metrics_rel
        results_json      = base_dir / results_json_rel
        print(f"Loading '{cname}' from {other_metrics_dir} ...")
        records, skipped = load_records(
            other_metrics_dir, results_json,
            skip_missing=args.skip_missing_results,
        )
        print(f"  Loaded {len(records)} examples, skipped {len(skipped)}")
        if skipped:
            for p, r in skipped[:3]:
                print(f"    skip: {Path(p).name} — {r}")
            if len(skipped) > 3:
                print(f"    … and {len(skipped) - 3} more")
        case_records.append((cname, records))
        case_names.append(cname)

    # ------------------------------------------------------------------
    # Evaluate each (k_spec, metric, version) combination
    # ------------------------------------------------------------------
    # results[k_spec][metric_name][version_name] = bottom_k_accuracy_dict
    all_results: dict = {}

    total_combos = len(k_specs) * len(metrics_to_run) * len(case_names)
    done = 0

    for k_spec in k_specs:
        kind = k_spec.partition(":")[0].strip().lower()
        offset = args.bocpd_offset if kind == "bocpd" else 0

        print(f"\n--- k_spec = {k_spec}  (offset={offset}) ---")
        all_results[k_spec] = {}

        for metric_name in metrics_to_run:
            all_results[k_spec][metric_name] = {}
            version_results: List[Tuple[str, dict]] = []

            for cname, records in case_records:
                # Compute first-k metric value per example
                scored: List[Tuple[float, int, str]] = []
                for rec in records:
                    m = compute_firstk_metrics(rec, k_spec, offset, bocpd_params)
                    val = m[metric_name]
                    if math.isfinite(val):
                        scored.append((val, rec.correct, rec.example_id))

                n_total = len(records)
                res = compute_bottom_k_accuracy(scored, percentiles)
                res["n_total"] = n_total

                all_results[k_spec][metric_name][cname] = res
                version_results.append((cname, res))
                done += 1

            print_table(k_spec, metric_name, version_results, percentiles)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    json_path = output_dir / "bottom_k_accuracy_firstk.json"
    csv_path  = output_dir / "bottom_k_accuracy_firstk.csv"

    save_json(all_results, json_path)
    save_csv(all_results, percentiles, case_names, csv_path)

    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
