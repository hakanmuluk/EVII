# EVII: Early Visual Information Integration

Official repository for the paper *Early Visual Information Integration (EVII)*.

---

## Abstract

In robotics, vision-language models (VLMs) support tasks that require perception-grounded reasoning, such as spatial understanding, action interpretation, and high-level planning. However, in multimodal chain-of-thought reasoning, it remains unclear when visual evidence is incorporated and how this relates to final answer reliability. We introduce Early Visual Information Integration (EVII), which measures the divergence between image-conditioned and no-image next-token distributions over an early reasoning prefix. EVII shows a stronger relationship with correctness than several widely used confidence-based baselines. We further show that high- and low-EVII examples differ meaningfully in accuracy, that much of the useful visual information is incorporated early, and that a bounded early-prefix variant using at most the first 40 generated tokens remains informative and can be used for inference-time decisions such as routing, fallback, or sample selection.

---

## Method

At each decoding step \(t\), EVII measures how much the next-token distribution changes when the image is removed. Concretely, it computes the Jensen--Shannon divergence between the image-conditioned distribution \(p_t^{\text{img}}\) and the no-image distribution \(p_t^{\text{no-img}}\), while keeping the generated prefix fixed.

\[
\mathrm{EVII}(k)
=
\frac{1}{k}
\sum_{t=1}^{k}
\mathrm{JSD}\!\left(p_t^{\text{img}} \,\|\, p_t^{\text{no-img}}\right)
\]

A larger divergence at step \(t\) means that the model's next-token distribution is more strongly affected by visual input. A smaller divergence suggests that the model is relying more on language-only information.

Rather than fixing *k* globally, we select it adaptively per example using Bayesian Online Changepoint Detection (BOCPD) with a Beta-family predictive model over the no-image JS signal, bounded to at most the first 40 generated tokens (*K*cap = 40). This identifies the point at which early visual influence begins to stabilise, yielding an example-specific integration horizon.

For inference-time use, BOCPD is run with fixed parameters (*k*min = 20, ERL = 30, *k*max = 40), allowing EVII to be computed from a bounded early-token budget before the full chain-of-thought is complete.

---

## Results

### Correlation with Answer Correctness

Weighted-binned Pearson correlation between each metric and answer correctness. Metric scores are first normalised to [0, 1] and grouped into bins of width 0.01. Higher is better. **Best per row in bold.**


| Dataset          | Log Prob   | Self-Certainty | Neg. Entropy | Neg. Perplexity | EVII (Ours) |
| ---------------- | ---------- | -------------- | ------------ | --------------- | ----------- |
| Robo2VLM (8B)    | 0.6938     | 0.7789         | 0.7039       | 0.7182          | **0.9040**  |
| Spatial-MM (8B)  | 0.6923     | 0.4869         | 0.6654       | 0.6556          | **0.7673**  |
| ERQA (8B)        | 0.1129     | 0.2222         | 0.1778       | 0.1852          | **0.5463**  |
| Robo2VLM (30B)   | 0.5269     | 0.6971         | 0.5032       | 0.4699          | **0.7370**  |
| Spatial-MM (30B) | **0.7391** | 0.6629         | 0.7444       | 0.7286          | 0.6246      |
| ERQA (30B)       | 0.4930     | 0.3673         | 0.4627       | 0.4156          | **0.5179**  |
| **Average**      | 0.5430     | 0.5359         | 0.5429       | 0.5288          | **0.6829**  |


### Selective Accuracy via EVII

Accuracy on examples ranked by EVII score. Lower is better for the lowest-ranked subsets; higher is better for the highest-ranked subsets.


| Version          | Overall Acc. | Lowest 10% | Lowest 30% | Highest 10% | Highest 30% |
| ---------------- | ------------ | ---------- | ---------- | ----------- | ----------- |
| Robo2VLM (8B)    | 55.64%       | 32.67%     | 39.87%     | **81.19%**  | 75.42%      |
| Spatial-MM (8B)  | 67.70%       | 52.00%     | 57.00%     | **88.00%**  | 79.00%      |
| ERQA (8B)        | 44.50%       | 27.50%     | 35.00%     | **57.50%**  | 52.50%      |
| Robo2VLM (30B)   | 65.93%       | 56.44%     | 52.49%     | **84.16%**  | 79.73%      |
| Spatial-MM (30B) | 69.30%       | 62.00%     | 63.00%     | **81.00%**  | 77.67%      |
| ERQA (30B)       | 44.75%       | 45.00%     | 35.83%     | **60.00%**  | 50.83%      |


### Inference-Time Reliability (Bounded Budget)

Average accuracy change relative to overall accuracy across all six benchmark–model settings. EVII uses BOCPD with *k*min=20, ERL=30, *k*max=40; baselines are computed on the first 40 tokens. Higher is better for top subsets; lower is better for bottom subsets.


| Metric              | Bottom 10%   | Bottom 30%   | Top 10%      | Top 30%      |
| ------------------- | ------------ | ------------ | ------------ | ------------ |
| **EVII**            | **−6.00 pp** | **−4.50 pp** | **+4.11 pp** | **+5.83 pp** |
| Log Probability     | −1.86 pp     | −0.13 pp     | +2.01 pp     | +0.52 pp     |
| Negative Entropy    | −1.61 pp     | −0.65 pp     | −0.54 pp     | +1.86 pp     |
| Negative Perplexity | −2.94 pp     | −1.43 pp     | +2.45 pp     | +2.19 pp     |
| Self-Certainty      | −1.34 pp     | −0.65 pp     | −0.78 pp     | −0.00 pp     |


---

## Experimental Setup

**Benchmarks:**

- [Robo2VLM](https://arxiv.org/abs/2505.15517) — 1,000-example subset (robot manipulation VQA)
- [Spatial-MM](https://arxiv.org/abs/2411.06048) — 1,000-example subset (spatial reasoning VQA)
- [ERQA](https://huggingface.co/datasets/FlagEval/ERQA) — 400-example benchmark (embodied reasoning VQA)

**Models:**

- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen/Qwen3-VL-30B-A3B-Thinking`

All responses are generated with deterministic decoding.

---

## Repository Structure

```
EVII/
├── README.md
├── data_collection/
│   ├── dataset_preparation/       # Jupyter notebooks: download benchmarks, build JSONL manifests
│   │   ├── erqa.ipynb
│   │   ├── robo2vlm.ipynb
│   │   └── spatial_mm.ipynb
│   └── inference/                 # Inference scripts: generate responses and compute EVII traces
│       ├── run_8b_single_image.py
│       ├── run_8b_multi_image.py
│       ├── run_30b.py
│       ├── run_8b_masked_after_k.py
│       ├── run_8b_other_metrics.py
│       └── run_30b_other_metrics.py
└── analysis/                      # Correlation, selective accuracy, and ablation scripts
    ├── metric_definition_dynamic.py
    ├── inference_metric_definition_dynamic.py
    ├── correlate_evii_metric.py
    ├── correlate_other_metrics.py
    ├── top_k_accuracy.py
    ├── bottom_k_accuracy.py
    ├── top_k_accuracy_other_metrics_firstk.py
    ├── bottom_k_accuracy_other_metrics_firstk.py
    ├── top_k_accuracy_inference.py
    └── bottom_k_accuracy_inference.py
```

---

## Requirements

Python 3.10+ and a CUDA-capable GPU (≥24 GB VRAM for 8B, ≥80 GB for 30B) are required.

```bash
pip install torch torchvision transformers accelerate datasets \
            qwen-vl-utils pillow tqdm requests
```

---

## Phase 1 — Data Collection

### Step 1: Prepare Datasets

Run the three notebooks in `data_collection/dataset_preparation/` to download benchmarks from HuggingFace, build formatted MCQ prompts, save images to disk, and write JSONL manifests.


| Notebook           | Output JSONL                   | Output images               |
| ------------------ | ------------------------------ | --------------------------- |
| `erqa.ipynb`       | `erqa_sample_400.jsonl`        | `erqa_sample_images/`       |
| `robo2vlm.ipynb`   | `robo2vlm_sample_951.jsonl`    | `robo2vlm_sample_images/`   |
| `spatial_mm.ipynb` | `spatial_mm_sample_1000.jsonl` | `spatial_mm_sample_images/` |


Each JSONL line contains `id`, `prompt`, and either `image_path` (single image) or `image_paths` (list, for ERQA).

> The notebooks also contain evaluation cells that score model outputs against ground-truth labels — those cells are part of the analysis phase and should be run after Step 2.

---

### Step 2: Run Inference

All inference scripts write one JSON file per example into `--output-dir`. Choose the script that matches the model size and dataset type.

#### Step 2a — 8B model, single-image datasets (Robo2VLM, Spatial-MM)

```bash
python data_collection/inference/run_8b_single_image.py \
  --input-jsonl <dataset>.jsonl \
  --batch-size 10 \
  --js-batch-size 2 \
  --analysis-chunk-size 8 \
  --max-new-tokens 4096 \
  --output-dir <output_dir>
```

Model: `Qwen/Qwen3-VL-8B-Thinking` (default). Each output JSON contains:

- `example` — input metadata
- `chat_result` — generated text and token ids
- `js_trace` — per-layer Jensen-Shannon divergence across generation steps
- `noimage_comparison_trace` — per-token JS between image-conditioned and text-only distributions

#### Step 2b — 8B model, multi-image dataset (ERQA)

```bash
python data_collection/inference/run_8b_multi_image.py \
  --input-jsonl <dataset>.jsonl \
  --batch-size 10 \
  --js-batch-size 2 \
  --analysis-chunk-size 8 \
  --max-new-tokens 4096 \
  --output-dir <output_dir>
```

Identical output schema to Step 2a; handles `image_paths` lists automatically.

#### Step 2c — 30B model (any dataset)

```bash
python data_collection/inference/run_30b.py \
  --input-jsonl <dataset>.jsonl \
  --output-dir <output_dir> \
  --batch-size 2 \
  --js-batch-size 2 \
  --analysis-chunk-size 8 \
  --max-new-tokens 4096
```

Model: `Qwen/Qwen3-VL-30B-A3B-Thinking` (default). Each output JSON contains:

- `example` — input metadata
- `chat_result` — generated text and token ids
- `js_trace` — per-layer Jensen-Shannon divergence across generation steps
- `noimage_comparison_trace` — per-token JS between image-conditioned and text-only distributions

#### Step 2d — 8B model, masked-image ablation (Robo2VLM, first 250 rows)

This experiment generates the first `k` tokens with full image access, then continues generation from the same KV cache while blocking all image-token positions in the attention mask.

```bash
python data_collection/inference/run_8b_masked_after_k.py \
  --model-name Qwen/Qwen3-VL-8B-Thinking \
  --input-jsonl <input_jsonl> \
  --output-dir <output_dir> \
  --max-new-tokens 4096 \
  --k-with-image <k>
```

Repeat with different `--k-with-image` values to sweep over masking points. Each output JSON contains:

- `example` — input metadata
- `chat_result` — full generated text
- `masked_continuation_trace` — per-token records indicating whether image positions were accessible, plus blocked position indices

To also block vision boundary tokens, add `--also-block-vision-boundary-tokens`.

#### Step 2e — 8B model, additional token metrics from existing outputs

Reuses pre-existing generation JSONs (produced by Steps 2a/2b) to compute per-token `log_prob`, `entropy`, `self_certainty`, and `no_image_js` without re-running generation.

```bash
python data_collection/inference/run_8b_other_metrics.py \
  --prefill-output-dir <existing_inference_output_dir> \
  --output-dir <output_dir> \
  --batch-size 2 \
  --analysis-chunk-size 8
```

#### Step 2f — 30B model, additional token metrics from existing outputs

Same reanalysis pass for 30B outputs (produced by Step 2c):

```bash
python data_collection/inference/run_30b_other_metrics.py \
  --prefill-output-dir <existing_inference_output_dir> \
  --output-dir <output_dir> \
  --batch-size 2 \
  --analysis-chunk-size 8
```

Each output JSON contains `token_metrics_trace` with per-token `log_prob`, `entropy`, `self_certainty`, and `no_image_js`.

---

## Output Directory Layout

After running inference, outputs follow this layout (example):

```
outputs_json/
└── <output_dir>/
    ├── example_0.json
    ├── example_1.json
    └── ...
```

---

## Phase 2 — Analysis

### Step 3a — Generate evaluation result JSONs

The dataset notebooks each contain an evaluation cell that scores model outputs against ground-truth labels and writes a results JSON. Run those cells after completing Phase 1 inference:

- `data_collection/dataset_preparation/erqa.ipynb` → e.g. `qwen3vl_8b_evaluation_results_erqa.json`
- `data_collection/dataset_preparation/robo2vlm.ipynb` → e.g. `qwen3vl_8b_evaluation_results_robo2vlm.json`
- `data_collection/dataset_preparation/spatial_mm.ipynb` → e.g. `qwen3vl_8b_evaluation_results_spatial_mm.json`

These files are required as `--results-jsons` inputs for all steps below.

---

### Step 3b — EVII metric correlation

Computes weighted-binned Pearson correlation between the dynamic EVII metric and per-example accuracy across a grid of BOCPD hyperparameters.

```bash
python analysis/correlate_evii_metric.py \
  --json-dirs \
    <infer_out_1>,<infer_out_2>,... \
  --results-jsons \
    <results_1>.json,<results_2>.json,... \
  --version-names \
    <name_1>,<name_2>,... \
  --output-dir <output_dir> \
  --offsets 0 \
  --adaptive-k-max-search-tokens 40 \
  --adaptive-k-cp-prob-thresholds 0.10 \
  --adaptive-k-drop-ratios 0.7 \
  --adaptive-k-beta-concentrations 20 \
  --ratio-bin-widths 0.01
```

---

### Step 3c — Other metrics correlation

Measures weighted-binned Pearson correlation for four scalar token metrics (avg log-prob, self-certainty, neg-entropy, neg-perplexity) vs. accuracy.

```bash
python analysis/correlate_other_metrics.py \
  --json-dirs <other_metrics_output_dir> \
  --results-jsons <results>.json \
  --version-names <version_name> \
  --output-dir <output_dir> \
  --bin-width 0.01
```

---

### Step 3d — Top-k / Bottom-k accuracy of EVII metric

Ranks examples by the EVII score (high = most confident) and reports accuracy in the top/bottom percentile slices.

```bash
python analysis/top_k_accuracy.py \
  --json-dirs \
    <infer_out_1>,<infer_out_2>,... \
  --other-metrics-json-dirs \
    <other_metrics_out_1>,<other_metrics_out_2>,... \
  --results-jsons \
    <results_1>.json,<results_2>.json,... \
  --version-names <name_1>,<name_2>,... \
  --output-dir <output_dir> \
  --adaptive-k-cp-prob-threshold 0.1 \
  --adaptive-k-drop-ratio 0.7 \
  --adaptive-k-beta-concentration 20.0 \
  --skip-missing-results
```

```bash
python analysis/bottom_k_accuracy.py \
  --json-dirs \
    <infer_out_1>,<infer_out_2>,... \
  --other-metrics-json-dirs \
    <other_metrics_out_1>,<other_metrics_out_2>,... \
  --results-jsons \
    <results_1>.json,<results_2>.json,... \
  --version-names <name_1>,<name_2>,... \
  --output-dir <output_dir> \
  --skip-missing-results
```

---

### Step 3e — Top-k / Bottom-k accuracy of other metrics (first-k tokens)

Computes the four token-level metrics (log-prob, self-certainty, neg-entropy, neg-perplexity) over only the first `k` tokens and reports top/bottom-percentile accuracy.

These scripts resolve input data directories relative to the `analysis/` folder (i.e., `Path(__file__).parent`), regardless of where the script is invoked. The six expected input directories are listed in the `CASES` constant inside each script (e.g. `erqa_8b_other_metrics`, `robo2vlm_8b_other_metrics`, `spatial_mm_8b_other_metrics`, and their 30B equivalents) and must therefore exist **inside `analysis/`**. The `--output-dir` path, by contrast, resolves relative to the current working directory, so running from the repo root will place output under `<repo_root>/<output_dir>`. Alternatively, pass `--base-dir <path>` to override the input base directory.

```bash
python analysis/top_k_accuracy_other_metrics_firstk.py \
  --k-specs "constant:40" \
  --percentiles "0.10" \
  --output-dir <output_dir> \
  --skip-missing-results
```

```bash
python analysis/bottom_k_accuracy_other_metrics_firstk.py \
  --k-specs "constant:40" \
  --percentiles "0.10" \
  --output-dir <output_dir> \
  --skip-missing-results
```

---

### Step 3f — Top-k / Bottom-k accuracy using inference metric

Same as Step 3d but uses the inference-time variant of the EVII metric (`inference_metric_definition_dynamic`).

```bash
python analysis/top_k_accuracy_inference.py \
  --json-dirs \
    <infer_out_1>,<infer_out_2>,... \
  --results-jsons \
    <results_1>.json,<results_2>.json,... \
  --other-metrics-json-dirs \
    <other_metrics_out_1>,<other_metrics_out_2>,... \
  --version-names <name_1>,<name_2>,... \
  --output-dir <output_dir> \
  --offset 0 \
  --skip-missing-results
```

```bash
python analysis/bottom_k_accuracy_inference.py \
  --json-dirs \
    <infer_out_1>,<infer_out_2>,... \
  --results-jsons \
    <results_1>.json,<results_2>.json,... \
  --other-metrics-json-dirs \
    <other_metrics_out_1>,<other_metrics_out_2>,... \
  --version-names <name_1>,<name_2>,... \
  --output-dir <output_dir> \
  --offset 0 \
  --skip-missing-results
```

---

