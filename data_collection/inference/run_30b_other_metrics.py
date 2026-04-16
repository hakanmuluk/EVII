
#!/usr/bin/env python3
"""
batched_qwen3_vl_thinking_js_30b_fixed.py

Batched two-stage Qwen3-VL-30B-A3B-Thinking analyzer.

What it does
------------
1. Loads one Qwen/Qwen3-VL-30B-A3B-Thinking model onto GPU.
2. Processes multiple examples in batches.
3. Pass 1: batched generation (with image).
4. Pass 2: batched teacher-forced analysis — two forward passes per batch:
     a) with image     → log_prob, entropy, self_certainty per generated token
     b) without image  → no_image_js (JS between with-image and text-only dists) per token
5. Saves one JSON file per example.
6. No plot generation.

Important fix in this version
-----------------------------
Teacher-forced analysis now extends token_type_ids / mm_token_type_ids alongside
input_ids and attention_mask when appending generated continuation tokens.
Without that, Qwen3-VL-MoE can crash with an IndexError caused by attention_mask
length not matching token-type tensor length.

Input options
-------------
A) Batched dataset via JSONL:
   each line should be one JSON object like:
   {"id": "ex1", "image_path": "/path/to/img1.jpg", "prompt": "Describe this image."}
   {"id": "ex2", "image_url": "https://...", "prompt": "What is happening here?"}

B) Single example via CLI:
   --image-path ... --prompt ...
   or
   --image-url ... --prompt ...

Notes
-----
- This uses plain batched inference in Transformers.
- For batching, tokenizer padding side is set to LEFT.
- One JSON file is written per example under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration


# =========================
# Dataclasses
# =========================

@dataclass
class ExampleInput:
    example_id: str
    prompt: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    resolved_image_path: Optional[str] = None
    image_paths: Optional[List[str]] = None
    resolved_image_paths: Optional[List[str]] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class ChatResult:
    model_name: str
    image_path: str
    prompt: str
    full_text: str
    thinking: str
    answer: str
    output_token_ids: List[int]
    prompt_token_count: int
    image_paths: Optional[List[str]] = None


@dataclass
class TokenMetricsTrace:
    """
    Per-token scalar metrics computed during teacher-forced analysis.

    log_probs[t]        = log P(selected token t | context, image)
    entropies[t]        = H(p_t) = -sum_v p_t(v) * log p_t(v)
    self_certainties[t] = KL(uniform || p_t)
                        = -log(V) - (1/V) * sum_v log p_t(v)
    no_image_js[t]      = JS(p_with_image_t || p_without_image_t)
    """
    token_ids: List[int]
    token_texts: List[str]
    log_probs: List[float]
    entropies: List[float]
    self_certainties: List[float]
    no_image_js: List[float]


@dataclass
class IntermediateResult:
    example: ExampleInput
    chat_result: ChatResult


# =========================
# Utilities
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunk_list(xs: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = name.strip("._")
    return name or "example"


def parse_torch_dtype(dtype_str: str):
    dtype_str = dtype_str.lower()
    if dtype_str == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_str}")
    return mapping[dtype_str]


def download_image(url: str, save_dir: Path, filename: Optional[str] = None) -> Path:
    ensure_dir(save_dir)

    if filename is None:
        filename = Path(url.split("?")[0]).name or "downloaded_image.jpg"

    save_path = save_dir / filename
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


def resolve_example_image(example: ExampleInput, download_dir: Path) -> ExampleInput:
    if example.image_path and example.image_url:
        raise ValueError(
            f"Example {example.example_id}: provide only one of image_path or image_url."
        )

    if example.image_path:
        p = Path(example.image_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Example {example.example_id}: image not found: {p}"
            )
        example.resolved_image_path = str(p)
        return example

    if example.image_url:
        filename = (
            f"{sanitize_filename(example.example_id)}_"
            f"{Path(example.image_url.split('?')[0]).name or 'downloaded.jpg'}"
        )
        p = download_image(example.image_url, save_dir=download_dir, filename=filename)
        example.resolved_image_path = str(p)
        return example

    raise ValueError(
        f"Example {example.example_id}: provide either image_path or image_url."
    )


def split_thinking_and_answer_from_text(full_text: str) -> Tuple[str, str]:
    text = full_text.strip()

    if "</think>" in text:
        left, right = text.split("</think>", 1)
        thinking = left.strip()
        answer = right.strip()

        if thinking.startswith("<think>"):
            thinking = thinking[len("<think>"):].strip()

        return thinking, answer

    if "<think>" in text:
        thinking = text.split("<think>", 1)[1].strip()
        return thinking, ""

    return "", text


def split_thinking_and_answer_from_ids(
    processor: AutoProcessor,
    output_ids: List[int],
) -> Tuple[str, str, str]:
    tokenizer = processor.tokenizer

    full_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    thinking, answer = split_thinking_and_answer_from_text(full_text)
    return full_text, thinking, answer


def safe_token_label(tokenizer, token_id: int) -> str:
    try:
        s = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        s = str(token_id)

    s = s.replace("\n", "\\n")
    if s == "":
        s = f"[{token_id}]"
    return s


def trim_right_padding(token_ids: List[int], pad_token_id: Optional[int]) -> List[int]:
    if pad_token_id is None:
        return token_ids

    trimmed = list(token_ids)
    while trimmed and trimmed[-1] == pad_token_id:
        trimmed.pop()
    return trimmed


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def compute_token_log_probs(
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    For each position, return log P of the actually selected token.
    probs:     (batch, seq_len, vocab)
    token_ids: (batch, seq_len)  — integer indices of the selected tokens
    returns:   (batch, seq_len)
    """
    safe_probs = torch.clamp(probs, min=eps)
    gathered = safe_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return torch.log(gathered)


def compute_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Shannon entropy H(p) = -sum_v p(v) * log p(v).
    probs:   (batch, seq_len, vocab)
    returns: (batch, seq_len)
    """
    safe_probs = torch.clamp(probs, min=eps)
    return -torch.sum(safe_probs * torch.log(safe_probs), dim=-1)


def compute_self_certainty(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(uniform || p_t) = -log(V) - (1/V) * sum_v log p_t(v).
    This equals log(V) - H(p_t) only when no clamping distorts things,
    but we implement the direct formula for correctness.
    probs:   (batch, seq_len, vocab)
    returns: (batch, seq_len)
    """
    V = probs.shape[-1]
    safe_probs = torch.clamp(probs, min=eps)
    mean_log_p = torch.mean(torch.log(safe_probs), dim=-1)
    return -math.log(V) - mean_log_p


# =========================
# I/O loading
# =========================

def load_examples_from_jsonl(jsonl_path: Path) -> List[ExampleInput]:
    examples: List[ExampleInput] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            example_id = obj.get("id", f"example_{idx:06d}")
            prompt = obj["prompt"]

            examples.append(
                ExampleInput(
                    example_id=example_id,
                    prompt=prompt,
                    image_path=obj.get("image_path"),
                    image_url=obj.get("image_url"),
                    image_paths=obj.get("image_paths"),
                )
            )

    return examples


def load_examples_from_args(args) -> List[ExampleInput]:
    if args.input_jsonl is not None:
        return load_examples_from_jsonl(Path(args.input_jsonl))

    if not args.prompt:
        raise ValueError("For single-example mode, --prompt is required.")

    if not args.image_path and not args.image_url:
        raise ValueError(
            "For single-example mode, provide one of --image-path or --image-url."
        )

    example_id = "example_000001"
    if args.image_path:
        example_id = sanitize_filename(Path(args.image_path).stem)

    return [
        ExampleInput(
            example_id=example_id,
            prompt=args.prompt,
            image_path=args.image_path,
            image_url=args.image_url,
        )
    ]


def load_intermediates_from_output_dir(dir_path: Path) -> List[IntermediateResult]:
    """
    Load pre-generated model outputs from a directory of JSON files produced by a
    previous run of this script (or a compatible generation script).

    Each JSON file must contain at minimum:
      - "example"    : dict with example_id, prompt, and image path fields
      - "chat_result": dict with output_token_ids and other generation fields

    Files that already contain a "token_metrics_trace" key are skipped (idempotency).

    Both single-image format (image_path / resolved_image_path) and multi-image
    format (image_paths / resolved_image_paths) are handled transparently.

    macOS often leaves AppleDouble resource-fork files named ``._*.json`` in folders;
    those are not valid JSON and are ignored without per-file warnings.
    """
    intermediates: List[IntermediateResult] = []
    all_json_paths = sorted(dir_path.glob("*.json"))
    appledouble_n = sum(1 for p in all_json_paths if p.name.startswith("._"))
    json_files = [p for p in all_json_paths if not p.name.startswith("._")]
    if appledouble_n:
        print(
            f"  Ignoring {appledouble_n} macOS ._*.json resource-fork file(s) (not real exports).",
            flush=True,
        )

    if not json_files:
        print(f"  Warning: no usable *.json files found in {dir_path}", flush=True)
        return intermediates

    skipped = 0
    for json_file in json_files:
        try:
            raw = json_file.read_bytes()
            if not raw.strip():
                print(f"  Warning: skipping {json_file.name} (empty file)", flush=True)
                skipped += 1
                continue
            text = raw.decode("utf-8", errors="replace")
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping {json_file.name} (JSON parse error: {e})", flush=True)
            skipped += 1
            continue

        if "token_metrics_trace" in payload:
            skipped += 1
            continue

        ex_dict = payload["example"]
        cr_dict = payload["chat_result"]

        if ex_dict.get("resolved_image_paths"):
            resolved_image_paths: List[str] = ex_dict["resolved_image_paths"]
            resolved_image_path: Optional[str] = None
        elif ex_dict.get("resolved_image_path"):
            resolved_image_path = ex_dict["resolved_image_path"]
            resolved_image_paths = [resolved_image_path]
        else:
            resolved_image_path = None
            resolved_image_paths = []

        example = ExampleInput(
            example_id=ex_dict["example_id"],
            prompt=ex_dict["prompt"],
            image_path=ex_dict.get("image_path"),
            image_url=ex_dict.get("image_url"),
            resolved_image_path=resolved_image_path,
            image_paths=ex_dict.get("image_paths"),
            resolved_image_paths=resolved_image_paths if resolved_image_paths else None,
        )

        chat_result = ChatResult(
            model_name=cr_dict.get("model_name", ""),
            image_path=cr_dict.get("image_path", ""),
            image_paths=cr_dict.get("image_paths"),
            prompt=cr_dict.get("prompt", ex_dict["prompt"]),
            full_text=cr_dict.get("full_text", ""),
            thinking=cr_dict.get("thinking", ""),
            answer=cr_dict.get("answer", ""),
            output_token_ids=cr_dict["output_token_ids"],
            prompt_token_count=cr_dict.get("prompt_token_count", 0),
        )

        intermediates.append(IntermediateResult(example=example, chat_result=chat_result))

    print(
        f"  Loaded {len(intermediates)} example(s) from {dir_path}"
        + (f" (skipped {skipped} file(s): empty/invalid JSON or already has metrics)" if skipped else ""),
        flush=True,
    )
    return intermediates


def load_intermediates_from_prefill_dirs(dirs: List[str]) -> List[IntermediateResult]:
    """Aggregate pre-generated outputs from multiple directories."""
    all_intermediates: List[IntermediateResult] = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            raise FileNotFoundError(f"Prefill output directory not found: {p}")
        print(f"Loading pre-generated outputs from: {p}", flush=True)
        all_intermediates.extend(load_intermediates_from_output_dir(p))
    print(f"Total pre-generated examples loaded: {len(all_intermediates)}", flush=True)
    return all_intermediates


def save_result_json(
    example: ExampleInput,
    chat_result: ChatResult,
    token_metrics_trace: TokenMetricsTrace,
    output_dir: Path,
) -> Path:
    ensure_dir(output_dir)

    filename = f"{sanitize_filename(example.example_id)}.json"
    save_path = output_dir / filename

    payload = {
        "example": asdict(example),
        "chat_result": asdict(chat_result),
        "token_metrics_trace": asdict(token_metrics_trace),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return save_path


# =========================
# Main analyzer
# =========================

class QwenVLThinkingAnalyzer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        parsed_dtype = parse_torch_dtype(torch_dtype)

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=parsed_dtype,
            device_map=device_map,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Important for batched decoder-style generation
        self.processor.tokenizer.padding_side = "left"
        if (
            self.processor.tokenizer.pad_token_id is None
            and self.processor.tokenizer.eos_token is not None
        ):
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Keep tokenizer/model pad ids aligned.
        if self.processor.tokenizer.pad_token_id is not None:
            if getattr(self.model.config, "pad_token_id", None) is None:
                self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
            if hasattr(self.model.config, "text_config") and getattr(self.model.config.text_config, "pad_token_id", None) is None:
                self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.input_device = next(self.model.parameters()).device

    def build_messages(
        self,
        image_paths: List[str],
        prompt: str,
        include_image: bool = True,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if include_image:
            for p in image_paths:
                content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _get_resolved_image_paths(ex: ExampleInput) -> List[str]:
        """Return a list of resolved image paths for an example, regardless of format."""
        if ex.resolved_image_paths:
            return ex.resolved_image_paths
        if ex.resolved_image_path:
            return [ex.resolved_image_path]
        return []

    def build_batch_inputs(
        self,
        examples: List[ExampleInput],
        include_image: bool = True,
    ) -> Dict[str, Any]:
        conversations = [
            self.build_messages(
                image_paths=self._get_resolved_image_paths(ex),
                prompt=ex.prompt,
                include_image=include_image,
            )
            for ex in examples
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        tensor_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                tensor_inputs[k] = v.to(self.input_device)
            else:
                tensor_inputs[k] = v

        return tensor_inputs

    @torch.inference_mode()
    def generate_responses_from_inputs(
        self,
        examples: List[ExampleInput],
        inputs: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[ChatResult]:
        if generation_config is None:
            generation_config = GenerationConfig()

        prompt_block_len = inputs["input_ids"].shape[1]
        prompt_token_counts = inputs["attention_mask"].sum(dim=1).tolist()

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
        )

        generated_only = generated_ids[:, prompt_block_len:]
        pad_token_id = self.processor.tokenizer.pad_token_id

        results: List[ChatResult] = []
        for i, ex in enumerate(examples):
            output_ids = trim_right_padding(generated_only[i].tolist(), pad_token_id)

            full_text, thinking, answer = split_thinking_and_answer_from_ids(
                self.processor,
                output_ids,
            )

            resolved_paths = self._get_resolved_image_paths(ex)
            results.append(
                ChatResult(
                    model_name=self.model_name,
                    image_path=resolved_paths[0] if len(resolved_paths) == 1 else (ex.resolved_image_path or ""),
                    image_paths=resolved_paths if len(resolved_paths) != 1 else None,
                    prompt=ex.prompt,
                    full_text=full_text,
                    thinking=thinking,
                    answer=answer,
                    output_token_ids=output_ids,
                    prompt_token_count=int(prompt_token_counts[i]),
                )
            )

        return results

    @staticmethod
    def _extend_optional_sequence_tensor(
        base_tensor: torch.Tensor,
        append_len: int,
        fill_value: int = 0,
    ) -> torch.Tensor:
        """
        Extend a (batch, seq_len) tensor on the right by append_len positions.
        """
        if append_len == 0:
            return base_tensor

        extension = torch.full(
            (base_tensor.shape[0], append_len),
            fill_value=fill_value,
            dtype=base_tensor.dtype,
            device=base_tensor.device,
        )
        return torch.cat([base_tensor, extension], dim=1)

    def _build_teacher_forced_inputs(
        self,
        inputs: Dict[str, Any],
        generated_token_ids_batch: List[List[int]],
    ) -> Tuple[Dict[str, Any], torch.Tensor, int, int]:
        """
        Append generated token IDs to prompt tensors for teacher-forced analysis.

        Returns:
            full_inputs, gen_attn, prompt_block_len, max_gen_len

        Important:
        - input_ids and attention_mask are extended with generated continuation.
        - token_type_ids / mm_token_type_ids are also extended with zeros because
          the continuation is plain text. This prevents shape mismatch crashes
          inside Qwen3-VL-MoE when attention_mask is applied to token-type tensors.
        """
        tokenizer = self.processor.tokenizer
        batch_size = len(generated_token_ids_batch)

        prompt_input_ids = inputs["input_ids"]
        prompt_attention_mask = inputs["attention_mask"]
        prompt_block_len = prompt_input_ids.shape[1]

        gen_lengths = [len(x) for x in generated_token_ids_batch]
        max_gen_len = max(gen_lengths) if gen_lengths else 0

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        gen_ids_tensor = torch.full(
            (batch_size, max_gen_len),
            fill_value=pad_token_id,
            dtype=prompt_input_ids.dtype,
            device=prompt_input_ids.device,
        )
        gen_attn = torch.zeros(
            (batch_size, max_gen_len),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )

        for i, ids in enumerate(generated_token_ids_batch):
            if not ids:
                continue
            gen_len = len(ids)
            gen_ids_tensor[i, :gen_len] = torch.tensor(
                ids,
                dtype=prompt_input_ids.dtype,
                device=prompt_input_ids.device,
            )
            gen_attn[i, :gen_len] = 1

        full_input_ids = torch.cat([prompt_input_ids, gen_ids_tensor], dim=1)
        full_attention_mask = torch.cat([prompt_attention_mask, gen_attn], dim=1)

        full_inputs: Dict[str, Any] = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        }

        # Extend token type tensors too; appended continuation is text.
        for type_key in ("token_type_ids", "mm_token_type_ids"):
            if type_key in inputs and torch.is_tensor(inputs[type_key]):
                full_inputs[type_key] = self._extend_optional_sequence_tensor(
                    inputs[type_key],
                    append_len=max_gen_len,
                    fill_value=0,
                )

        # Carry over all remaining non-sequence / multimodal tensors unchanged.
        for k, v in inputs.items():
            if k not in {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}:
                full_inputs[k] = v

        return full_inputs, gen_attn, prompt_block_len, max_gen_len

    @torch.inference_mode()
    def compute_token_metrics_from_inputs(
        self,
        image_inputs: Dict[str, Any],
        text_inputs: Dict[str, Any],
        generated_token_ids_batch: List[List[int]],
        analysis_chunk_size: int = 8,
    ) -> List[TokenMetricsTrace]:
        """
        Run two teacher-forced passes (with image, without image) and compute
        per-token metrics for each generated token:
          - log_prob        : log P(selected token | context, image)
          - entropy         : H(p_t)
          - self_certainty  : KL(uniform || p_t)
          - no_image_js     : JS(p_with_image || p_without_image)
        """
        batch_size = len(generated_token_ids_batch)
        tokenizer = self.processor.tokenizer

        if batch_size == 0:
            return []

        if all(len(x) == 0 for x in generated_token_ids_batch):
            return [
                TokenMetricsTrace(
                    token_ids=[],
                    token_texts=[],
                    log_probs=[],
                    entropies=[],
                    self_certainties=[],
                    no_image_js=[],
                )
                for _ in range(batch_size)
            ]

        (
            img_full_inputs,
            gen_attn,
            img_prompt_block_len,
            max_gen_len,
        ) = self._build_teacher_forced_inputs(image_inputs, generated_token_ids_batch)

        (
            txt_full_inputs,
            _,
            txt_prompt_block_len,
            _,
        ) = self._build_teacher_forced_inputs(text_inputs, generated_token_ids_batch)

        if img_full_inputs["input_ids"].shape[1] != img_full_inputs["attention_mask"].shape[1]:
            raise RuntimeError("WITH-image teacher-forced inputs have mismatched input_ids vs attention_mask length.")
        if txt_full_inputs["input_ids"].shape[1] != txt_full_inputs["attention_mask"].shape[1]:
            raise RuntimeError("WITHOUT-image teacher-forced inputs have mismatched input_ids vs attention_mask length.")

        for type_key in ("token_type_ids", "mm_token_type_ids"):
            if type_key in img_full_inputs and torch.is_tensor(img_full_inputs[type_key]):
                if img_full_inputs[type_key].shape[1] != img_full_inputs["attention_mask"].shape[1]:
                    raise RuntimeError(
                        f"WITH-image teacher-forced {type_key} length mismatch: "
                        f"{img_full_inputs[type_key].shape[1]} vs attention_mask {img_full_inputs['attention_mask'].shape[1]}"
                    )
            if type_key in txt_full_inputs and torch.is_tensor(txt_full_inputs[type_key]):
                if txt_full_inputs[type_key].shape[1] != txt_full_inputs["attention_mask"].shape[1]:
                    raise RuntimeError(
                        f"WITHOUT-image teacher-forced {type_key} length mismatch: "
                        f"{txt_full_inputs[type_key].shape[1]} vs attention_mask {txt_full_inputs['attention_mask'].shape[1]}"
                    )

        print("  Running teacher-forced pass WITH image...", flush=True)
        image_outputs = self.model(
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **img_full_inputs,
        )

        print("  Running teacher-forced pass WITHOUT image...", flush=True)
        text_outputs = self.model(
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **txt_full_inputs,
        )

        token_texts_batch = [
            [safe_token_label(tokenizer, tid) for tid in ids]
            for ids in generated_token_ids_batch
        ]

        log_probs_batch: List[List[float]] = [[] for _ in range(batch_size)]
        entropies_batch: List[List[float]] = [[] for _ in range(batch_size)]
        self_cert_batch: List[List[float]] = [[] for _ in range(batch_size)]
        noimage_js_batch: List[List[float]] = [[] for _ in range(batch_size)]

        print(
            f"  Computing token metrics: batch_size={batch_size}, "
            f"max_gen_len={max_gen_len}, analysis_chunk_size={analysis_chunk_size}",
            flush=True,
        )

        for start in range(0, max_gen_len, analysis_chunk_size):
            end = min(start + analysis_chunk_size, max_gen_len)

            img_pos = list(
                range(img_prompt_block_len - 1 + start, img_prompt_block_len - 1 + end)
            )
            txt_pos = list(
                range(txt_prompt_block_len - 1 + start, txt_prompt_block_len - 1 + end)
            )

            print(f"    Token chunk {start}:{end}", flush=True)

            valid_mask = gen_attn[:, start:end].bool()
            valid_mask_cpu = valid_mask.detach().cpu()

            img_logits = image_outputs.logits[:, img_pos, :].float()
            img_probs = F.softmax(img_logits, dim=-1)

            txt_logits = text_outputs.logits[:, txt_pos, :].float()
            txt_probs = F.softmax(txt_logits, dim=-1)

            selected_ids = img_full_inputs["input_ids"][
                :, img_prompt_block_len + start : img_prompt_block_len + end
            ]

            lp_vals = compute_token_log_probs(img_probs, selected_ids)
            ent_vals = compute_entropy(img_probs)
            sc_vals = compute_self_certainty(img_probs)
            js_vals = js_divergence(img_probs, txt_probs)

            lp_cpu = lp_vals.detach().cpu()
            ent_cpu = ent_vals.detach().cpu()
            sc_cpu = sc_vals.detach().cpu()
            js_cpu = js_vals.detach().cpu()

            for b in range(batch_size):
                vmask = valid_mask_cpu[b]
                log_probs_batch[b].extend(lp_cpu[b][vmask].tolist())
                entropies_batch[b].extend(ent_cpu[b][vmask].tolist())
                self_cert_batch[b].extend(sc_cpu[b][vmask].tolist())
                noimage_js_batch[b].extend(js_cpu[b][vmask].tolist())

            del (
                img_logits, img_probs,
                txt_logits, txt_probs,
                selected_ids,
                lp_vals, ent_vals, sc_vals, js_vals,
                lp_cpu, ent_cpu, sc_cpu, js_cpu,
                valid_mask, valid_mask_cpu,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        traces: List[TokenMetricsTrace] = []
        for b in range(batch_size):
            traces.append(
                TokenMetricsTrace(
                    token_ids=generated_token_ids_batch[b],
                    token_texts=token_texts_batch[b],
                    log_probs=log_probs_batch[b],
                    entropies=entropies_batch[b],
                    self_certainties=self_cert_batch[b],
                    no_image_js=noimage_js_batch[b],
                )
            )

        del image_outputs, text_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return traces


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batched Qwen3-VL-30B-A3B-Thinking: generation + per-token metrics analysis"
    )

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")

    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="JSONL manifest, one example per line."
    )

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--image-url", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--download-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs_json")

    parser.add_argument(
        "--prefill-output-dir",
        type=str,
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "Directory containing pre-generated output JSON files. "
            "When given, Pass 1 (generation) is skipped and the script only runs "
            "the teacher-forced metric analysis (Pass 2). "
            "Can be repeated to load from multiple directories, e.g. "
            "--prefill-output-dir spatial_mm_qwen3vl_30b_outputs "
            "--prefill-output-dir erqa_qwen3vl_30b_outputs"
        ),
    )

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--analysis-chunk-size", type=int, default=8)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    return parser


def main(cli_args=None) -> None:
    parser = build_parser()
    args, _ = parser.parse_known_args(cli_args)

    download_dir = Path(args.download_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(download_dir)
    ensure_dir(output_dir)

    analyzer = QwenVLThinkingAnalyzer(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    intermediates: List[IntermediateResult] = []

    if args.prefill_output_dir:
        print("\n=== PREFILL MODE: Loading pre-generated outputs ===", flush=True)
        intermediates = load_intermediates_from_prefill_dirs(args.prefill_output_dir)

        if not intermediates:
            print("No examples to process (all already processed or no files found). Exiting.", flush=True)
            return

    else:
        examples = load_examples_from_args(args)

        print(f"Loaded {len(examples)} example(s).", flush=True)
        print("Resolving image paths / downloading URLs...", flush=True)

        resolved_examples: List[ExampleInput] = []
        for ex in examples:
            resolved_examples.append(resolve_example_image(ex, download_dir=download_dir))

        gen_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        generation_batches = chunk_list(resolved_examples, args.batch_size)

        print("\n=== PASS 1: Generation ===", flush=True)

        for batch_idx, batch_examples in enumerate(generation_batches, start=1):
            print(
                f"\n[Generation] Batch {batch_idx}/{len(generation_batches)} | "
                f"size={len(batch_examples)}",
                flush=True,
            )

            print("Building batch inputs WITH image...", flush=True)
            inputs = analyzer.build_batch_inputs(batch_examples, include_image=True)

            print("Batched generation...", flush=True)
            chat_results = analyzer.generate_responses_from_inputs(
                batch_examples,
                inputs,
                generation_config=gen_cfg,
            )

            for ex, chat_result in zip(batch_examples, chat_results):
                intermediates.append(
                    IntermediateResult(
                        example=ex,
                        chat_result=chat_result,
                    )
                )

            del inputs, chat_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    analysis_batches = chunk_list(intermediates, args.batch_size)

    print("\n=== PASS 2: Token Metrics Analysis ===", flush=True)

    total_saved = 0
    for batch_idx, batch_items in enumerate(analysis_batches, start=1):
        print(
            f"\n[Analysis] Batch {batch_idx}/{len(analysis_batches)} | "
            f"size={len(batch_items)}",
            flush=True,
        )

        batch_examples = [x.example for x in batch_items]
        batch_chat_results = [x.chat_result for x in batch_items]
        generated_token_ids_batch = [x.output_token_ids for x in batch_chat_results]

        print("Rebuilding batch inputs WITH image...", flush=True)
        image_inputs = analyzer.build_batch_inputs(batch_examples, include_image=True)

        print("Building batch inputs WITHOUT image...", flush=True)
        text_inputs = analyzer.build_batch_inputs(batch_examples, include_image=False)

        print("Running token metrics analysis...", flush=True)
        token_metrics_traces = analyzer.compute_token_metrics_from_inputs(
            image_inputs=image_inputs,
            text_inputs=text_inputs,
            generated_token_ids_batch=generated_token_ids_batch,
            analysis_chunk_size=args.analysis_chunk_size,
        )

        print("Saving per-example JSON files...", flush=True)
        for ex, chat_result, trace in zip(
            batch_examples, batch_chat_results, token_metrics_traces
        ):
            save_path = save_result_json(
                example=ex,
                chat_result=chat_result,
                token_metrics_trace=trace,
                output_dir=output_dir,
            )
            total_saved += 1
            print(f"  Saved: {save_path}", flush=True)

        del image_inputs, text_inputs, token_metrics_traces
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Saved {total_saved} JSON file(s) to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
