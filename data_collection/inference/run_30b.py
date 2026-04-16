#!/usr/bin/env python3
"""
batched_qwen3_vl_30b_a3b_thinking_js.py

Batched three-stage Qwen3-VL-30B-A3B-Thinking analyzer.

What it does
------------
1. Loads one Qwen/Qwen3-VL-30B-A3B-Thinking model onto GPU.
2. Processes multiple examples in batches.
3. Pass 1: batched generation with image(s)
4. Pass 2: batched teacher-forced forward pass with image(s)
   - computes per-layer JS divergence against the final image-conditioned distribution
5. Pass 3: batched teacher-forced forward pass with NO image
   - uses the same prompt text and the same generated continuation from Pass 1
   - computes per-token JS divergence between:
       final(with image) vs final(no image)
6. Saves one JSON file per example
7. No plot generation

Important notes
---------------
- Uses Qwen3VLMoeForConditionalGeneration for the 30B-A3B-Thinking checkpoint.
- Extends token_type_ids / mm_token_type_ids alongside input_ids and attention_mask
  when appending generated continuation tokens to prevent shape mismatch crashes
  inside Qwen3-VL-MoE.
- Supports both single-image (image_path / image_url) and multi-image
  (image_paths / image_urls) examples.

Input options
-------------
A) Batched dataset via JSONL:
   each line should be one JSON object like:
   {"id": "ex1", "image_path": "/path/to/img1.jpg", "prompt": "Describe this image."}
   {"id": "ex2", "image_url": "https://...", "prompt": "What is happening here?"}
   {"id": "ex3", "image_paths": ["/path/img1.jpg", "/path/img2.jpg"], "prompt": "Compare."}

B) Single example via CLI:
   --image-path ... --prompt ...
   or
   --image-url ... --prompt ...

Notes
-----
- This uses plain batched inference in Transformers.
- For batching, tokenizer padding side is set to LEFT.
- One JSON file is written per example under --output-dir.
- Generation and analysis can use different batch sizes.
- Pass 2 + Pass 3 together use more memory than a single pass.
  If you hit OOM, reduce --js-batch-size first.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, Qwen3VLMoeForConditionalGeneration


# =========================
# Dataclasses
# =========================

@dataclass
class ExampleInput:
    example_id: str
    prompt: str

    # Single-image inputs
    image_path: Optional[str] = None
    image_url: Optional[str] = None

    # Multi-image inputs
    image_paths: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None

    # Resolved local paths
    resolved_image_path: Optional[str] = None
    resolved_image_paths: List[str] = field(default_factory=list)


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
    image_paths: List[str]
    prompt: str
    full_text: str
    thinking: str
    answer: str
    output_token_ids: List[int]
    prompt_token_count: int


@dataclass
class JSDivergenceTrace:
    """
    js_matrix[layer_index][token_index] = JS(layer_i || final_image_conditioned)
    for that generated token.
    """
    token_ids: List[int]
    token_texts: List[str]
    js_matrix: List[List[float]]
    mean_js_per_layer: List[float]


@dataclass
class TokenJSValue:
    token_index: int
    token_id: int
    token_text: str
    js_divergence: float


@dataclass
class NoImageComparisonTrace:
    """
    js_per_token[token_index] = JS(final_with_image || final_without_image)
    for that generated token.
    """
    reference_condition: str
    comparison_condition: str
    token_ids: List[int]
    token_texts: List[str]
    js_per_token: List[float]
    mean_js: float
    token_js_pairs: List[TokenJSValue]


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
    dtype_str = str(dtype_str).lower()
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


def _normalize_optional_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        out = [str(x) for x in value if x is not None and str(x) != ""]
        return out if len(out) > 0 else None
    raise TypeError(f"Expected list or None, got {type(value)}")


def _validate_exactly_one_image_source(example: ExampleInput) -> None:
    provided = []

    if example.image_path is not None:
        provided.append("image_path")
    if example.image_url is not None:
        provided.append("image_url")
    if example.image_paths is not None:
        provided.append("image_paths")
    if example.image_urls is not None:
        provided.append("image_urls")

    if len(provided) != 1:
        raise ValueError(
            f"Example {example.example_id}: provide exactly one of "
            f"image_path, image_url, image_paths, or image_urls. Got: {provided}"
        )


def resolve_example_image(example: ExampleInput, download_dir: Path) -> ExampleInput:
    _validate_exactly_one_image_source(example)

    resolved: List[str] = []

    if example.image_path is not None:
        p = Path(example.image_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Example {example.example_id}: image not found: {p}"
            )
        resolved = [str(p)]

    elif example.image_paths is not None:
        for img_path in example.image_paths:
            p = Path(img_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Example {example.example_id}: image not found: {p}"
                )
            resolved.append(str(p))

        if len(resolved) == 0:
            raise ValueError(
                f"Example {example.example_id}: image_paths is empty."
            )

    elif example.image_url is not None:
        filename = (
            f"{sanitize_filename(example.example_id)}_"
            f"{Path(example.image_url.split('?')[0]).name or 'downloaded.jpg'}"
        )
        p = download_image(example.image_url, save_dir=download_dir, filename=filename)
        resolved = [str(p)]

    elif example.image_urls is not None:
        if len(example.image_urls) == 0:
            raise ValueError(
                f"Example {example.example_id}: image_urls is empty."
            )

        for idx, url in enumerate(example.image_urls):
            filename = (
                f"{sanitize_filename(example.example_id)}_{idx:03d}_"
                f"{Path(url.split('?')[0]).name or 'downloaded.jpg'}"
            )
            p = download_image(url, save_dir=download_dir, filename=filename)
            resolved.append(str(p))

    else:
        raise ValueError(
            f"Example {example.example_id}: provide an image source."
        )

    example.resolved_image_paths = resolved
    example.resolved_image_path = resolved[0] if len(resolved) > 0 else None
    return example


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

    s = s.replace(" ", "␠").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
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


def _resolve_attr_path(root: Any, path: Sequence[str]) -> Optional[Any]:
    obj = root
    for attr in path:
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def get_module_device(module) -> Optional[torch.device]:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return None


def maybe_apply_final_norm(
    model: Qwen3VLMoeForConditionalGeneration,
    hidden: torch.Tensor,
) -> torch.Tensor:
    candidate_paths = [
        ["model", "language_model", "norm"],
        ["model", "norm"],
        ["language_model", "model", "norm"],
        ["language_model", "norm"],
        ["text_model", "norm"],
    ]

    for path in candidate_paths:
        obj = _resolve_attr_path(model, path)
        if obj is not None and callable(obj):
            try:
                return obj(hidden)
            except RuntimeError:
                dev = get_module_device(obj)
                if dev is not None and hidden.device != dev:
                    return obj(hidden.to(dev))
                raise

    raise RuntimeError(
        "Could not find a final norm module on the model. "
        f"Tried paths: {candidate_paths}"
    )


def resolve_lm_head(model: Qwen3VLMoeForConditionalGeneration):
    candidate_paths = [
        ["lm_head"],
        ["model", "lm_head"],
        ["language_model", "lm_head"],
        ["model", "language_model", "lm_head"],
        ["text_model", "lm_head"],
    ]

    for path in candidate_paths:
        obj = _resolve_attr_path(model, path)
        if obj is not None and callable(obj):
            return obj

    raise AttributeError("Could not find lm_head on the provided model.")


def extract_hidden_states_from_outputs(outputs: Any):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states
    if hasattr(outputs, "language_model_outputs"):
        lmo = outputs.language_model_outputs
        if hasattr(lmo, "hidden_states") and lmo.hidden_states is not None:
            return lmo.hidden_states
    raise RuntimeError(
        "Model outputs did not contain hidden_states. "
        "Make sure output_hidden_states=True is supported."
    )


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
            example_id = str(obj.get("id", f"example_{idx:06d}"))
            prompt = str(obj["prompt"])

            image_path = obj.get("image_path")
            image_url = obj.get("image_url")
            image_paths = _normalize_optional_str_list(obj.get("image_paths"))
            image_urls = _normalize_optional_str_list(obj.get("image_urls"))

            examples.append(
                ExampleInput(
                    example_id=example_id,
                    prompt=prompt,
                    image_path=str(image_path) if image_path is not None else None,
                    image_url=str(image_url) if image_url is not None else None,
                    image_paths=image_paths,
                    image_urls=image_urls,
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


def save_result_json(
    example: ExampleInput,
    chat_result: ChatResult,
    js_trace: JSDivergenceTrace,
    noimage_comparison_trace: NoImageComparisonTrace,
    output_dir: Path,
) -> Path:
    ensure_dir(output_dir)

    filename = f"{sanitize_filename(example.example_id)}.json"
    save_path = output_dir / filename

    payload = {
        "example": asdict(example),
        "chat_result": asdict(chat_result),
        "js_trace": asdict(js_trace),
        "noimage_comparison_trace": asdict(noimage_comparison_trace),
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
        attn_implementation: Optional[str] = None,
    ):
        self.model_name = model_name
        parsed_dtype = parse_torch_dtype(torch_dtype)

        # Load config first and patch missing pad_token_id on buggy builds.
        config = AutoConfig.from_pretrained(model_name)
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None and not hasattr(text_cfg, "pad_token_id"):
            outer_pad = getattr(config, "pad_token_id", None)
            eos_id = getattr(config, "eos_token_id", None)
            if isinstance(eos_id, (list, tuple)):
                eos_id = eos_id[0] if len(eos_id) > 0 else None
            patched_pad = outer_pad if outer_pad is not None else eos_id
            setattr(text_cfg, "pad_token_id", patched_pad)

        model_kwargs: Dict[str, Any] = {
            "config": config,
            "torch_dtype": parsed_dtype,
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

        if (
            self.processor.tokenizer.pad_token_id is None
            and self.processor.tokenizer.eos_token is not None
        ):
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Keep config aligned with tokenizer pad token.
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        if hasattr(self.model.config, "text_config"):
            if getattr(self.model.config.text_config, "pad_token_id", None) is None:
                self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id

        self.input_device = next(self.model.parameters()).device
        self.lm_head = resolve_lm_head(self.model)
        self.lm_head_device = get_module_device(self.lm_head)

    def build_messages(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        include_image: bool = True,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        if include_image:
            if not image_paths or len(image_paths) == 0:
                raise ValueError("include_image=True but image_paths is missing or empty.")
            for image_path in image_paths:
                content.append({"type": "image", "image": image_path})

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    def build_batch_inputs(
        self,
        examples: List[ExampleInput],
        include_image: bool = True,
    ) -> Dict[str, Any]:
        conversations = [
            self.build_messages(
                prompt=ex.prompt,
                image_paths=ex.resolved_image_paths,
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

        tensor_inputs = {}
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

        generate_kwargs = dict(
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        if generation_config.do_sample:
            generate_kwargs.update(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
            )

        generated_ids = self.model.generate(
            **inputs,
            **generate_kwargs,
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

            results.append(
                ChatResult(
                    model_name=self.model_name,
                    image_path=ex.resolved_image_path or "",
                    image_paths=list(ex.resolved_image_paths),
                    prompt=ex.prompt,
                    full_text=full_text,
                    thinking=thinking,
                    answer=answer,
                    output_token_ids=output_ids,
                    prompt_token_count=int(prompt_token_counts[i]),
                )
            )

        return results

    def prepare_teacher_forced_inputs(
        self,
        inputs: Dict[str, Any],
        generated_token_ids_batch: List[List[int]],
    ) -> Tuple[Dict[str, Any], torch.Tensor, int, int]:
        """
        Append generated token IDs to prompt tensors for teacher-forced analysis.

        Returns:
            full_inputs, gen_attn, prompt_block_len, max_gen_len

        Extends token_type_ids / mm_token_type_ids alongside input_ids and
        attention_mask because Qwen3-VL-MoE uses them for multimodal bookkeeping;
        a shape mismatch here causes an IndexError during the forward pass.
        """
        tokenizer = self.processor.tokenizer

        prompt_input_ids = inputs["input_ids"]
        prompt_attention_mask = inputs["attention_mask"]
        prompt_block_len = prompt_input_ids.shape[1]

        batch_size = len(generated_token_ids_batch)
        gen_lengths = [len(x) for x in generated_token_ids_batch]
        max_gen_len = max(gen_lengths) if len(gen_lengths) > 0 else 0

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

        static_kwargs = {
            k: v for k, v in inputs.items()
            if k not in {"input_ids", "attention_mask"}
        }

        # Generated continuation tokens are plain text — extend with zeros.
        for mm_key in ("mm_token_type_ids", "token_type_ids"):
            if mm_key in static_kwargs and torch.is_tensor(static_kwargs[mm_key]):
                mm_ids = static_kwargs[mm_key]
                mm_pad = torch.zeros(
                    (mm_ids.shape[0], max_gen_len),
                    dtype=mm_ids.dtype,
                    device=mm_ids.device,
                )
                static_kwargs[mm_key] = torch.cat([mm_ids, mm_pad], dim=1)

        full_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            **static_kwargs,
        }

        return full_inputs, gen_attn, prompt_block_len, max_gen_len

    @torch.inference_mode()
    def compute_stage2_and_stage3_from_inputs(
        self,
        image_inputs: Dict[str, Any],
        text_only_inputs: Dict[str, Any],
        generated_token_ids_batch: List[List[int]],
        analysis_chunk_size: int = 8,
    ) -> Tuple[List[JSDivergenceTrace], List[NoImageComparisonTrace]]:
        batch_size = len(generated_token_ids_batch)
        tokenizer = self.processor.tokenizer

        if batch_size == 0:
            return [], []

        token_texts_batch = [
            [safe_token_label(tokenizer, tid) for tid in ids]
            for ids in generated_token_ids_batch
        ]

        if all(len(x) == 0 for x in generated_token_ids_batch):
            empty_js = [
                JSDivergenceTrace(
                    token_ids=[],
                    token_texts=[],
                    js_matrix=[],
                    mean_js_per_layer=[],
                )
                for _ in range(batch_size)
            ]
            empty_noimage = [
                NoImageComparisonTrace(
                    reference_condition="with_image",
                    comparison_condition="without_image_text_only",
                    token_ids=[],
                    token_texts=[],
                    js_per_token=[],
                    mean_js=0.0,
                    token_js_pairs=[],
                )
                for _ in range(batch_size)
            ]
            return empty_js, empty_noimage

        image_full_inputs, gen_attn, image_prompt_block_len, max_gen_len = (
            self.prepare_teacher_forced_inputs(
                inputs=image_inputs,
                generated_token_ids_batch=generated_token_ids_batch,
            )
        )

        text_full_inputs, text_gen_attn, text_prompt_block_len, text_max_gen_len = (
            self.prepare_teacher_forced_inputs(
                inputs=text_only_inputs,
                generated_token_ids_batch=generated_token_ids_batch,
            )
        )

        if text_max_gen_len != max_gen_len:
            raise RuntimeError(
                f"Internal error: generated lengths mismatch between image and text-only "
                f"teacher-forced inputs ({max_gen_len} vs {text_max_gen_len})."
            )

        if gen_attn.shape != text_gen_attn.shape:
            raise RuntimeError(
                f"Internal error: generated attention masks mismatch between image and text-only "
                f"teacher-forced inputs ({tuple(gen_attn.shape)} vs {tuple(text_gen_attn.shape)})."
            )

        print("  Running batched teacher-forced analysis pass WITH image(s)...", flush=True)
        image_outputs = self.model(
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **image_full_inputs,
        )

        print("  Running batched teacher-forced analysis pass WITHOUT image...", flush=True)
        text_outputs = self.model(
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **text_full_inputs,
        )

        hidden_states = extract_hidden_states_from_outputs(image_outputs)[1:]
        num_layers = len(hidden_states)

        js_matrix_batch: List[List[List[float]]] = [
            [[] for _ in range(num_layers)] for _ in range(batch_size)
        ]
        noimage_js_batch: List[List[float]] = [[] for _ in range(batch_size)]

        print(
            f"  Computing Stage 2 + Stage 3 metrics for batch_size={batch_size}, "
            f"max_gen_len={max_gen_len}, num_layers={num_layers}, "
            f"analysis_chunk_size={analysis_chunk_size}",
            flush=True,
        )

        last_layer_index = num_layers - 1

        for start in range(0, max_gen_len, analysis_chunk_size):
            end = min(start + analysis_chunk_size, max_gen_len)

            image_pos_chunk = list(
                range(image_prompt_block_len - 1 + start, image_prompt_block_len - 1 + end)
            )
            text_pos_chunk = list(
                range(text_prompt_block_len - 1 + start, text_prompt_block_len - 1 + end)
            )

            print(f"    Token chunk {start}:{end}", flush=True)

            valid_mask = gen_attn[:, start:end].bool()
            valid_mask_cpu = valid_mask.detach().cpu()

            image_final_logits_chunk = image_outputs.logits[:, image_pos_chunk, :].float()
            image_final_probs_chunk = F.softmax(image_final_logits_chunk, dim=-1)

            text_final_logits_chunk = text_outputs.logits[:, text_pos_chunk, :].float()
            text_final_probs_chunk = F.softmax(text_final_logits_chunk, dim=-1)

            noimage_js_vals = js_divergence(image_final_probs_chunk, text_final_probs_chunk)
            noimage_js_vals_cpu = noimage_js_vals.detach().cpu()

            for b in range(batch_size):
                valid_vals = noimage_js_vals_cpu[b][valid_mask_cpu[b]].tolist()
                noimage_js_batch[b].extend(valid_vals)

            for layer_idx, layer_hidden in enumerate(hidden_states):
                selected_hidden = layer_hidden[:, image_pos_chunk, :]
                if layer_idx != last_layer_index:
                    selected_hidden = maybe_apply_final_norm(self.model, selected_hidden)

                if self.lm_head_device is not None and selected_hidden.device != self.lm_head_device:
                    selected_hidden = selected_hidden.to(self.lm_head_device)

                layer_logits_chunk = self.lm_head(selected_hidden).float()
                if layer_logits_chunk.device != image_final_probs_chunk.device:
                    layer_logits_chunk = layer_logits_chunk.to(image_final_probs_chunk.device)

                layer_probs_chunk = F.softmax(layer_logits_chunk, dim=-1)
                js_vals = js_divergence(layer_probs_chunk, image_final_probs_chunk)
                js_vals_cpu = js_vals.detach().cpu()

                for b in range(batch_size):
                    valid_vals = js_vals_cpu[b][valid_mask_cpu[b]].tolist()
                    js_matrix_batch[b][layer_idx].extend(valid_vals)

                del selected_hidden, layer_logits_chunk, layer_probs_chunk, js_vals, js_vals_cpu

            del (
                image_final_logits_chunk,
                image_final_probs_chunk,
                text_final_logits_chunk,
                text_final_probs_chunk,
                noimage_js_vals,
                noimage_js_vals_cpu,
                valid_mask,
                valid_mask_cpu,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        js_traces: List[JSDivergenceTrace] = []
        noimage_traces: List[NoImageComparisonTrace] = []

        for b in range(batch_size):
            row_matrix = js_matrix_batch[b]
            mean_js_per_layer = [
                (sum(row) / len(row)) if len(row) > 0 else 0.0
                for row in row_matrix
            ]

            js_traces.append(
                JSDivergenceTrace(
                    token_ids=generated_token_ids_batch[b],
                    token_texts=token_texts_batch[b],
                    js_matrix=row_matrix,
                    mean_js_per_layer=mean_js_per_layer,
                )
            )

            js_per_token = noimage_js_batch[b]
            token_js_pairs = [
                TokenJSValue(
                    token_index=i,
                    token_id=generated_token_ids_batch[b][i],
                    token_text=token_texts_batch[b][i],
                    js_divergence=js_per_token[i],
                )
                for i in range(len(js_per_token))
            ]

            mean_js = (sum(js_per_token) / len(js_per_token)) if len(js_per_token) > 0 else 0.0

            noimage_traces.append(
                NoImageComparisonTrace(
                    reference_condition="with_image",
                    comparison_condition="without_image_text_only",
                    token_ids=generated_token_ids_batch[b],
                    token_texts=token_texts_batch[b],
                    js_per_token=js_per_token,
                    mean_js=mean_js,
                    token_js_pairs=token_js_pairs,
                )
            )

        del image_outputs, text_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return js_traces, noimage_traces


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batched Qwen3-VL-30B-A3B-Thinking + three-stage JS divergence analysis"
    )

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default=None)

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

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--js-batch-size", type=int, default=None)
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

    analyzer = QwenVLThinkingAnalyzer(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    js_batch_size = args.js_batch_size or args.batch_size

    # =========================
    # Pass 1: Generation
    # =========================
    generation_batches = chunk_list(resolved_examples, args.batch_size)
    intermediates: List[IntermediateResult] = []

    print("\n=== PASS 1: Generation ===", flush=True)

    for batch_idx, batch_examples in enumerate(generation_batches, start=1):
        print(
            f"\n[Generation] Batch {batch_idx}/{len(generation_batches)} | "
            f"size={len(batch_examples)}",
            flush=True,
        )

        print("Stage 1: Building batch inputs WITH image(s)...", flush=True)
        inputs = analyzer.build_batch_inputs(batch_examples, include_image=True)

        print("Stage 1: Batched generation...", flush=True)
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

    # =========================
    # Pass 2 + Pass 3: Analysis
    # =========================
    analysis_batches = chunk_list(intermediates, js_batch_size)

    print("\n=== PASS 2 + PASS 3: Analysis ===", flush=True)

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

        print("Stage 2: Rebuilding batch inputs WITH image(s)...", flush=True)
        image_inputs = analyzer.build_batch_inputs(batch_examples, include_image=True)

        print("Stage 3: Building batch inputs WITHOUT image...", flush=True)
        text_only_inputs = analyzer.build_batch_inputs(batch_examples, include_image=False)

        print("Stage 2 + Stage 3: Running batched analysis...", flush=True)
        js_traces, noimage_traces = analyzer.compute_stage2_and_stage3_from_inputs(
            image_inputs=image_inputs,
            text_only_inputs=text_only_inputs,
            generated_token_ids_batch=generated_token_ids_batch,
            analysis_chunk_size=args.analysis_chunk_size,
        )

        print("Saving per-example JSON files...", flush=True)
        for ex, chat_result, js_trace, noimage_trace in zip(
            batch_examples,
            batch_chat_results,
            js_traces,
            noimage_traces,
        ):
            save_path = save_result_json(
                example=ex,
                chat_result=chat_result,
                js_trace=js_trace,
                noimage_comparison_trace=noimage_trace,
                output_dir=output_dir,
            )
            total_saved += 1
            print(f"  Saved: {save_path}", flush=True)

        del image_inputs, text_only_inputs, js_traces, noimage_traces
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Saved {total_saved} JSON file(s) to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
