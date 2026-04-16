"""
Dynamic metric definition module.

This is the ONLY file that needs editing to change which per-example metric
is correlated against accuracy. The static runner imports:

    METRIC_NAME                  – human-readable label used in plots / logs
    compute_metric_for_example() – returns a dict with at least
                                   "metric_value" (float) and "correct" (int).

Grid-search ranges (deep_js_thresholds, critical_layers, vision_js_thresholds,
ratio_bin_widths) are supplied via the static runner's CLI, not here.

Change-point method: Beta-family BOCPD with adaptive window parameters
-----------------------------------------------------------------------
Identical to metric_definition_dynamic.py except for the window rules:

    total_tokens < 200  (short sequences — dynamic proportional windows)
        k_min               = int(total_tokens * 0.05)
        expected_run_length = int(total_tokens * 0.1)
        k_max               = int(total_tokens * 0.15)

    total_tokens >= 200  (long sequences — fixed windows)
        k_min               = 20
        expected_run_length = 30
        k_max               = 40
"""

import math
import numpy as np


METRIC_NAME = "avg_noimage_js_until_bocpd_beta_adaptive_stop"


def _build_shifted_noimage(noimage_js, total_tokens, vision_offset):
    aligned = np.full(total_tokens, np.nan, dtype=float)
    for i in range(total_tokens):
        j = i + vision_offset
        if 0 <= j < len(noimage_js):
            aligned[i] = noimage_js[j]
    return aligned


def _smooth_1d(x):
    """Very light smoothing to reduce single-token noise."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    return np.convolve(x, kernel, mode="same")


def _clip_unit_interval(x, eps=1e-6):
    return float(np.clip(float(x), eps, 1.0 - eps))


def _beta_pdf(x, alpha, beta, eps=1e-12):
    """
    Scalar Beta pdf, computed in log-space for stability.
    x must be in (0, 1).
    """
    x = _clip_unit_interval(x, eps=max(eps, 1e-12))
    alpha = max(float(alpha), eps)
    beta = max(float(beta), eps)

    log_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log1p(-x) - log_norm

    if log_pdf < -745.0:
        return 0.0
    return math.exp(log_pdf)


def _bocpd_run_length_posterior_beta(
    x,
    hazard=1.0 / 20.0,
    prior_mean=None,
    prior_kappa=1.0,
    beta_concentration=20.0,
    clip_eps=1e-6,
):
    """
    Approximate beta-family BOCPD.

    Observation model:
        x_t | m_r ~ Beta(m_r * phi, (1 - m_r) * phi)

    where:
        m_r  = current regime mean estimate for run length r
        phi  = beta_concentration

    Regime mean estimate uses prior shrinkage:
        m_r = (prior_kappa * prior_mean + sum_x_r) / (prior_kappa + count_r)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)

    if T == 0:
        return np.zeros((1, 1), dtype=float), np.zeros(0, dtype=float)

    x = np.clip(x, clip_eps, 1.0 - clip_eps)

    if prior_mean is None:
        prior_mean = float(np.mean(x))
    prior_mean = _clip_unit_interval(prior_mean, eps=clip_eps)

    prior_kappa = float(max(prior_kappa, 1e-6))
    beta_concentration = float(max(beta_concentration, 1e-3))
    hazard = float(np.clip(hazard, 1e-6, 1.0 - 1e-6))

    R = np.zeros((T + 1, T + 1), dtype=float)
    R[0, 0] = 1.0

    count_params = np.array([0.0], dtype=float)
    sum_params   = np.array([0.0], dtype=float)

    cp_probs = []

    for t in range(1, T + 1):
        xt = float(x[t - 1])

        prev_rl_count = t
        pred_probs = np.zeros(prev_rl_count, dtype=float)

        for r in range(prev_rl_count):
            count_r = count_params[r]
            sum_r   = sum_params[r]

            post_mean_r = (prior_kappa * prior_mean + sum_r) / (prior_kappa + count_r)
            post_mean_r = _clip_unit_interval(post_mean_r, eps=clip_eps)

            alpha_r = max(post_mean_r * beta_concentration, clip_eps)
            beta_r  = max((1.0 - post_mean_r) * beta_concentration, clip_eps)

            pred_probs[r] = _beta_pdf(xt, alpha_r, beta_r, eps=clip_eps)

        growth = R[t - 1, :prev_rl_count] * pred_probs * (1.0 - hazard)
        cp     = np.sum(R[t - 1, :prev_rl_count] * pred_probs * hazard)

        joint = np.zeros(t + 1, dtype=float)
        joint[0]  = cp
        joint[1:] = growth

        evidence = np.sum(joint)
        if evidence <= 1e-300:
            joint[:] = 1.0 / len(joint)
            evidence  = 1.0

        R[t, : t + 1] = joint / evidence
        cp_probs.append(float(R[t, 0]))

        new_count_params = np.empty(t + 1, dtype=float)
        new_sum_params   = np.empty(t + 1, dtype=float)

        new_count_params[0] = 0.0
        new_sum_params[0]   = 0.0

        for r in range(1, t + 1):
            new_count_params[r] = count_params[r - 1] + 1.0
            new_sum_params[r]   = sum_params[r - 1] + xt

        count_params = new_count_params
        sum_params   = new_sum_params

    return R, np.asarray(cp_probs, dtype=float)


def _choose_k_bocpd_beta(
    aligned_noimage_js,
    total_tokens,
    max_search_tokens=40,
    smooth=True,
    prior_mean=None,
    prior_kappa=1.0,
    beta_concentration=20.0,
    cp_prob_threshold=0.20,
    pre_window=5,
    post_window=4,
    drop_ratio=0.65,
    clip_eps=1e-6,
):
    """
    Choose k using beta-family BOCPD, then require a persistent downward transition.

    Window parameters are selected adaptively:
        total_tokens < 200:
            k_min               = int(total_tokens * 0.05)
            expected_run_length = int(total_tokens * 0.1)
            k_max               = int(total_tokens * 0.15)
        total_tokens >= 200:
            k_min               = 20
            expected_run_length = 30
            k_max               = 40

    Accept the earliest t in [k_min, search_end] such that:
      1) changepoint probability at t >= cp_prob_threshold
      2) mean(post-window after t) <= drop_ratio * mean(pre-window before t)

    If no such t exists, fall back to the argmax changepoint probability
    inside the search window.
    """
    total_tokens = int(total_tokens)
    if total_tokens <= 0:
        return 0, {}

    if total_tokens < 0:
        k_min               = int(total_tokens * 0.05)
        expected_run_length = int(total_tokens * 0.1)
        k_max               = int(total_tokens * 0.15)
    else:
        k_min               = 20
        expected_run_length = 30
        k_max               = 40

    expected_run_length = max(expected_run_length, 1)

    search_end = min(total_tokens, int(max_search_tokens), int(k_max))
    if search_end <= 0:
        return 0, {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "expected_run_length": int(expected_run_length),
        }

    x = np.asarray(aligned_noimage_js[:search_end], dtype=float)
    x = np.nan_to_num(x, nan=0.0)
    x = np.clip(x, clip_eps, 1.0 - clip_eps)

    if smooth:
        x = _smooth_1d(x)
        x = np.clip(x, clip_eps, 1.0 - clip_eps)

    if len(x) == 0:
        return 0, {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "expected_run_length": int(expected_run_length),
        }

    hazard = 1.0 / float(expected_run_length)

    _, cp_probs = _bocpd_run_length_posterior_beta(
        x,
        hazard=hazard,
        prior_mean=prior_mean,
        prior_kappa=prior_kappa,
        beta_concentration=beta_concentration,
        clip_eps=clip_eps,
    )

    lower = max(1, int(k_min))
    upper = min(int(search_end), int(k_max))

    if lower > upper:
        chosen_k = max(0, upper)
        return chosen_k, {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "expected_run_length": int(expected_run_length),
            "search_end": int(search_end),
            "selection_reason": "search_window_too_short",
            "max_cp_prob_in_window": float(cp_probs[:search_end].max()) if len(cp_probs) > 0 else float("nan"),
        }

    # Earliest acceptable persistent downward changepoint
    for t in range(lower, upper + 1):
        cp_prob_t = float(cp_probs[t - 1])

        pre_start = max(0, t - pre_window)
        pre       = x[pre_start:t]
        post_end  = min(search_end, t + post_window)
        post      = x[t:post_end]

        pre_mean  = float(np.mean(pre))  if len(pre)  > 0 else 0.0
        post_mean = float(np.mean(post)) if len(post) > 0 else 0.0

        downward = post_mean <= drop_ratio * (pre_mean + 1e-12)

        if cp_prob_t >= cp_prob_threshold and downward:
            return t, {
                "k_min": int(k_min),
                "k_max": int(k_max),
                "expected_run_length": int(expected_run_length),
                "search_end": int(search_end),
                "cp_prob_at_k": cp_prob_t,
                "pre_mean_at_k": pre_mean,
                "post_mean_after_k": post_mean,
                "selection_reason": "first_persistent_downward_changepoint",
                "cp_prob_threshold": float(cp_prob_threshold),
                "drop_ratio": float(drop_ratio),
                "pre_window": int(pre_window),
                "post_window": int(post_window),
                "beta_concentration": float(beta_concentration),
                "prior_kappa": float(prior_kappa),
            }

    # Fallback: choose the strongest changepoint inside the window
    window_cp = cp_probs[lower - 1 : upper]
    best_idx  = int(np.argmax(window_cp))
    chosen_k  = lower + best_idx

    pre_start = max(0, chosen_k - pre_window)
    pre       = x[pre_start:chosen_k]
    post_end  = min(search_end, chosen_k + post_window)
    post      = x[chosen_k:post_end]

    return chosen_k, {
        "k_min": int(k_min),
        "k_max": int(k_max),
        "expected_run_length": int(expected_run_length),
        "search_end": int(search_end),
        "cp_prob_at_k": float(cp_probs[chosen_k - 1]),
        "pre_mean_at_k": float(np.mean(pre))  if len(pre)  > 0 else 0.0,
        "post_mean_after_k": float(np.mean(post)) if len(post) > 0 else 0.0,
        "selection_reason": "fallback_max_cp_prob_in_window",
        "cp_prob_threshold": float(cp_prob_threshold),
        "drop_ratio": float(drop_ratio),
        "pre_window": int(pre_window),
        "post_window": int(post_window),
        "beta_concentration": float(beta_concentration),
        "prior_kappa": float(prior_kappa),
    }


def compute_metric_for_example(record, offset, **params):
    """
    Metric:
        average aligned no_image_js over tokens [0 : chosen_k)

    chosen_k is selected example-wise by beta-family BOCPD from the aligned
    no_image_js signal itself.

    Missing aligned positions (NaN due to offset going out of range)
    are treated as 0 contribution before clipping into (0,1).
    """
    shifted_noimage_js = _build_shifted_noimage(
        record.noimage_js,
        total_tokens=record.total_tokens,
        vision_offset=offset,
    )

    total_tokens = int(record.total_tokens)

    max_search_tokens   = int(params.get("adaptive_k_max_search_tokens", 40))
    smooth              = bool(params.get("adaptive_k_smooth", True))

    prior_mean = params.get("adaptive_k_prior_mean", None)
    if prior_mean is not None:
        prior_mean = float(prior_mean)

    prior_kappa         = float(params.get("adaptive_k_prior_kappa", 1.0))
    beta_concentration  = float(params.get("adaptive_k_beta_concentration", 20.0))
    cp_prob_threshold   = float(params.get("adaptive_k_cp_prob_threshold", 0.10))
    pre_window          = int(params.get("adaptive_k_pre_window", 5))
    post_window         = int(params.get("adaptive_k_post_window", 4))
    drop_ratio          = float(params.get("adaptive_k_drop_ratio", 0.7))
    clip_eps            = float(params.get("adaptive_k_clip_eps", 1e-6))

    chosen_k, k_debug = _choose_k_bocpd_beta(
        aligned_noimage_js=shifted_noimage_js,
        total_tokens=total_tokens,
        max_search_tokens=max_search_tokens,
        smooth=smooth,
        prior_mean=prior_mean,
        prior_kappa=prior_kappa,
        beta_concentration=beta_concentration,
        cp_prob_threshold=cp_prob_threshold,
        pre_window=pre_window,
        post_window=post_window,
        drop_ratio=drop_ratio,
        clip_eps=clip_eps,
    )

    chosen_k = int(min(max(chosen_k, 0), max_search_tokens))
    first_k_noimage_js = np.nan_to_num(shifted_noimage_js[:chosen_k], nan=0.0)

    noimage_js_sum_first_k = float(np.sum(first_k_noimage_js))
    denominator = int(chosen_k)

    metric_value = (
        noimage_js_sum_first_k / denominator
        if denominator > 0
        else float("nan")
    )

    return {
        "metric_value": metric_value,
        "correct": int(record.correct),
        "example_id": record.example_id,
        "total_tokens": total_tokens,
        "chosen_k": int(chosen_k),
        "tokens_used_in_sum": int(chosen_k),
        "denominator": int(denominator),
        "noimage_js_sum_first_k": noimage_js_sum_first_k,
        "valid_noimage_positions_first_k": int(np.isfinite(shifted_noimage_js[:chosen_k]).sum()),
        "adaptive_k_max_search_tokens": int(max_search_tokens),
        "adaptive_k_smooth": bool(smooth),
        "adaptive_k_prior_kappa": float(prior_kappa),
        "adaptive_k_beta_concentration": float(beta_concentration),
        "adaptive_k_cp_prob_threshold": float(cp_prob_threshold),
        "adaptive_k_pre_window": int(pre_window),
        "adaptive_k_post_window": int(post_window),
        "adaptive_k_drop_ratio": float(drop_ratio),
        "adaptive_k_clip_eps": float(clip_eps),
        **k_debug,
    }
