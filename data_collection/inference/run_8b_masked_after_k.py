#!/usr/bin/env python3
"""
qwen3_vl_mask_image_after_k.py

Generate with Qwen/Qwen3-VL-8B-Thinking while keeping the first k generated
tokens fully image-conditioned, then continuing generation from the SAME KV
cache but with the ORIGINAL image-token positions blocked from attention.

What this script does
---------------------
- Runs a normal multimodal prefill on the prompt + image.
- Generates the first k tokens normally.
- Keeps the resulting past_key_values (KV cache).
- For token k and later, continues decoding from that same cache, but passes a
  custom 4D attention mask that blocks the original image-token positions.
- Earlier generated text tokens remain visible, so they can carry image-derived
  information forward indirectly.

Important notes
---------------
- This script intentionally uses attn_implementation="eager".
- It processes examples sequentially (one example at a time) for reliability.
- It is written for Qwen3-VL-style multimodal causal decoding in Hugging Face.
- It assumes one image per example.
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
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    k_with_image: int = 16
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class TokenStepRecord:
    token_index: int
    token_id: int
    token_text: str
    used_image_access: bool


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


@dataclass
class MaskedContinuationTrace:
    k_with_image: int
    total_generated_tokens: int
    image_token_id: int
    vision_start_token_id: Optional[int]
    vision_end_token_id: Optional[int]
    also_block_vision_boundary_tokens: bool
    image_token_positions_in_prompt: List[int]
    blocked_prompt_positions_after_switch: List[int]
    token_records: List[TokenStepRecord]


# =========================
# Utilities
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    In-place style filtering on a 1D logits tensor.
    """
    logits = logits.clone()

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, filter_value), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = torch.where(remove_mask, torch.full_like(logits, filter_value), logits)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    """
    logits: shape [vocab_size]
    """
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 when sampling, got {temperature}")

    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)

    if not torch.isfinite(probs).all():
        raise RuntimeError("Non-finite probabilities encountered during sampling.")

    token_id = torch.multinomial(probs, num_samples=1)
    return int(token_id.item())


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
    masked_trace: MaskedContinuationTrace,
    output_dir: Path,
) -> Path:
    ensure_dir(output_dir)

    filename = f"{sanitize_filename(example.example_id)}.json"
    save_path = output_dir / filename

    payload = {
        "example": asdict(example),
        "chat_result": asdict(chat_result),
        "masked_continuation_trace": asdict(masked_trace),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return save_path


# =========================
# Main analyzer
# =========================

class QwenVLMaskedContinuationAnalyzer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Thinking",
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_name = model_name
        parsed_dtype = parse_torch_dtype(torch_dtype)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=parsed_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        if self.processor.tokenizer.pad_token_id is None and self.processor.tokenizer.eos_token is not None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # The input tensors should go to the device of the first parameter shard.
        self.input_device = next(self.model.parameters()).device

        self.image_token_id = int(getattr(self.model.config, "image_token_id"))
        self.vision_start_token_id = getattr(self.model.config, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(self.model.config, "vision_end_token_id", None)

    def build_messages(
        self,
        prompt: str,
        image_path: str,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def build_inputs_for_example(
        self,
        example: ExampleInput,
    ) -> Dict[str, Any]:
        if not example.resolved_image_path:
            raise ValueError("resolved_image_path is missing.")

        messages = self.build_messages(
            prompt=example.prompt,
            image_path=example.resolved_image_path,
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image = Image.open(example.resolved_image_path).convert("RGB")
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        tensor_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                tensor_inputs[k] = v.to(self.input_device)
            else:
                tensor_inputs[k] = v

        return tensor_inputs

    def get_blocked_prompt_positions(
        self,
        input_ids: torch.Tensor,
        also_block_vision_boundary_tokens: bool,
    ) -> List[int]:
        """
        input_ids: shape [1, seq_len]
        Returns prompt positions that should be blocked after the switch.
        """
        ids = input_ids[0].tolist()
        blocked: List[int] = []

        for pos, tid in enumerate(ids):
            if tid == self.image_token_id:
                blocked.append(pos)
            elif also_block_vision_boundary_tokens and self.vision_start_token_id is not None and tid == self.vision_start_token_id:
                blocked.append(pos)
            elif also_block_vision_boundary_tokens and self.vision_end_token_id is not None and tid == self.vision_end_token_id:
                blocked.append(pos)

        return blocked


    def build_masked_incremental_attention_mask(
        self,
        total_context_len: int,
        blocked_prompt_positions: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a 4D ADDITIVE attention mask for a single incremental decoding step.

        Shape: [1, 1, 1, total_context_len]
        - allowed positions get 0.0
        - blocked positions get a large negative value

        This is for eager attention, where the attention mask is added to the
        attention logits before softmax.
        """
        mask = torch.zeros((1, 1, 1, total_context_len), dtype=torch.float32, device=device)

        if blocked_prompt_positions:
            block_idx = torch.tensor(blocked_prompt_positions, dtype=torch.long, device=device)
            mask[:, :, :, block_idx] = torch.finfo(torch.float32).min

        return mask


    def prepare_prefill_model_inputs(

        self,
        full_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]

        cache_position = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )

        prepared = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
            **{k: v for k, v in full_inputs.items() if k not in {"input_ids", "attention_mask"}},
        )
        return prepared

    def prepare_incremental_model_inputs(
        self,
        next_input_ids: torch.Tensor,
        full_attention_mask_2d: torch.Tensor,
        past_key_values,
        rope_deltas,
        cache_position: torch.Tensor,
        blocked_prompt_positions: List[int],
        block_image_tokens_now: bool,
    ) -> Dict[str, Any]:
        prepared = self.model.prepare_inputs_for_generation(
            input_ids=next_input_ids,
            attention_mask=full_attention_mask_2d,
            past_key_values=past_key_values,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            use_cache=True,
        )

        if block_image_tokens_now:
            custom_mask_4d = self.build_masked_incremental_attention_mask(
                total_context_len=int(full_attention_mask_2d.shape[1]),
                blocked_prompt_positions=blocked_prompt_positions,
                device=next_input_ids.device,
            )
            prepared["attention_mask"] = custom_mask_4d

        return prepared


    @torch.inference_mode()
    def generate_with_image_then_masked_continuation(
        self,
        example: ExampleInput,
        generation_config: GenerationConfig,
        also_block_vision_boundary_tokens: bool = False,
    ) -> Tuple[ChatResult, MaskedContinuationTrace]:
        if generation_config.k_with_image < 1:
            raise ValueError(
                f"k_with_image must be >= 1 and refers ONLY to generated-token count, got {generation_config.k_with_image}"
            )

        full_inputs = self.build_inputs_for_example(example)
        prompt_input_ids = full_inputs["input_ids"]
        prompt_attention_mask = full_inputs["attention_mask"]

        if prompt_input_ids.shape[0] != 1:
            raise RuntimeError("This implementation expects batch size 1 per example.")

        prompt_token_count = int(prompt_attention_mask.sum(dim=1).item())
        blocked_prompt_positions = self.get_blocked_prompt_positions(
            input_ids=prompt_input_ids,
            also_block_vision_boundary_tokens=also_block_vision_boundary_tokens,
        )

        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id

        # ------------------------------------------------------------------
        # Phase 1: generate the first k GENERATED tokens with the NORMAL path.
        # This is the key fix: no custom mask and no custom incremental loop yet.
        # ------------------------------------------------------------------
        first_phase_len = min(generation_config.k_with_image, generation_config.max_new_tokens)

        normal_gen = self.model.generate(
            **full_inputs,
            max_new_tokens=first_phase_len,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        sequences = normal_gen.sequences
        prompt_block_len = prompt_input_ids.shape[1]
        first_phase_ids = sequences[0, prompt_block_len:].tolist()

        # Remove any trailing pad tokens from generate output.
        first_phase_ids = trim_right_padding(first_phase_ids, pad_token_id)

        generated_token_ids: List[int] = list(first_phase_ids)
        token_records: List[TokenStepRecord] = [
            TokenStepRecord(
                token_index=i,
                token_id=tid,
                token_text=safe_token_label(self.processor.tokenizer, tid),
                used_image_access=True,
            )
            for i, tid in enumerate(generated_token_ids)
        ]

        # If generation ended early or we already reached the requested total length,
        # there is no masked continuation to do.
        ended_in_first_phase = (
            len(generated_token_ids) == 0
            or (eos_token_id is not None and generated_token_ids[-1] == eos_token_id)
            or len(generated_token_ids) >= generation_config.max_new_tokens
            or len(generated_token_ids) < generation_config.k_with_image
        )

        if ended_in_first_phase:
            trimmed_ids = trim_right_padding(generated_token_ids, pad_token_id)
            full_text, thinking, answer = split_thinking_and_answer_from_ids(
                self.processor,
                trimmed_ids,
            )
            chat_result = ChatResult(
                model_name=self.model_name,
                image_path=example.resolved_image_path or "",
                prompt=example.prompt,
                full_text=full_text,
                thinking=thinking,
                answer=answer,
                output_token_ids=trimmed_ids,
                prompt_token_count=prompt_token_count,
            )
            trace = MaskedContinuationTrace(
                k_with_image=generation_config.k_with_image,
                total_generated_tokens=len(trimmed_ids),
                image_token_id=self.image_token_id,
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
                also_block_vision_boundary_tokens=also_block_vision_boundary_tokens,
                image_token_positions_in_prompt=[
                    i for i, tid in enumerate(prompt_input_ids[0].tolist()) if tid == self.image_token_id
                ],
                blocked_prompt_positions_after_switch=blocked_prompt_positions,
                token_records=token_records[:len(trimmed_ids)],
            )
            return chat_result, trace

        # ------------------------------------------------------------------
        # Phase 2 setup: rebuild cache normally up to the token BEFORE the switch.
        #
        # If the first k generated tokens are x0..x{k-1}, then token x{k} should be
        # generated WITHOUT direct image access. To do that correctly, we rebuild a
        # cache for [prompt + x0..x{k-2}], then feed x{k-1} with the masked attention
        # rule to obtain logits for x{k}.
        # ------------------------------------------------------------------
        prefix_ids_for_cache = generated_token_ids[:-1]
        current_input_token_id = generated_token_ids[-1]

        if len(prefix_ids_for_cache) > 0:
            prefix_ids_tensor = torch.tensor(
                [prefix_ids_for_cache],
                dtype=prompt_input_ids.dtype,
                device=prompt_input_ids.device,
            )
            prefix_attn = torch.ones(
                (1, len(prefix_ids_for_cache)),
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            )
            cache_build_input_ids = torch.cat([prompt_input_ids, prefix_ids_tensor], dim=1)
            cache_build_attention_mask = torch.cat([prompt_attention_mask, prefix_attn], dim=1)
        else:
            cache_build_input_ids = prompt_input_ids
            cache_build_attention_mask = prompt_attention_mask

        cache_build_inputs = {
            "input_ids": cache_build_input_ids,
            "attention_mask": cache_build_attention_mask,
            **{k: v for k, v in full_inputs.items() if k not in {"input_ids", "attention_mask"}},
        }

        prefill_inputs = self.prepare_prefill_model_inputs(cache_build_inputs)
        outputs = self.model(
            **prefill_inputs,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        rope_deltas = getattr(outputs, "rope_deltas", None)

        # For the first masked step, the token being fed is x{k-1}, and the visible
        # context length includes prompt + all first-k generated tokens.
        next_input_ids = torch.tensor(
            [[current_input_token_id]],
            dtype=prompt_input_ids.dtype,
            device=prompt_input_ids.device,
        )
        full_attention_mask_2d = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    (1, len(generated_token_ids)),
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )

        # ------------------------------------------------------------------
        # Phase 2: masked continuation
        # ------------------------------------------------------------------
        while len(generated_token_ids) < generation_config.max_new_tokens:
            generated_count_so_far = len(generated_token_ids)
            block_image_tokens_now = generated_count_so_far >= generation_config.k_with_image

            current_cache_position = torch.tensor(
                [prompt_input_ids.shape[1] + generated_count_so_far - 1],
                dtype=torch.long,
                device=prompt_input_ids.device,
            )

            step_inputs = self.prepare_incremental_model_inputs(
                next_input_ids=next_input_ids,
                full_attention_mask_2d=full_attention_mask_2d,
                past_key_values=past_key_values,
                rope_deltas=rope_deltas,
                cache_position=current_cache_position,
                blocked_prompt_positions=blocked_prompt_positions,
                block_image_tokens_now=block_image_tokens_now,
            )

            outputs = self.model(
                **step_inputs,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            if getattr(outputs, "rope_deltas", None) is not None:
                rope_deltas = outputs.rope_deltas

            logits = outputs.logits[0, -1, :]
            next_token_id = sample_next_token(
                logits=logits,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
            )

            token_records.append(
                TokenStepRecord(
                    token_index=len(generated_token_ids),
                    token_id=next_token_id,
                    token_text=safe_token_label(self.processor.tokenizer, next_token_id),
                    used_image_access=not block_image_tokens_now,
                )
            )
            generated_token_ids.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            next_input_ids = torch.tensor(
                [[next_token_id]],
                dtype=prompt_input_ids.dtype,
                device=prompt_input_ids.device,
            )
            full_attention_mask_2d = torch.cat(
                [
                    full_attention_mask_2d,
                    torch.ones((1, 1), dtype=full_attention_mask_2d.dtype, device=full_attention_mask_2d.device),
                ],
                dim=1,
            )

        trimmed_ids = trim_right_padding(generated_token_ids, pad_token_id)
        full_text, thinking, answer = split_thinking_and_answer_from_ids(
            self.processor,
            trimmed_ids,
        )

        chat_result = ChatResult(
            model_name=self.model_name,
            image_path=example.resolved_image_path or "",
            prompt=example.prompt,
            full_text=full_text,
            thinking=thinking,
            answer=answer,
            output_token_ids=trimmed_ids,
            prompt_token_count=prompt_token_count,
        )

        trace = MaskedContinuationTrace(
            k_with_image=generation_config.k_with_image,
            total_generated_tokens=len(trimmed_ids),
            image_token_id=self.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            also_block_vision_boundary_tokens=also_block_vision_boundary_tokens,
            image_token_positions_in_prompt=[
                i for i, tid in enumerate(prompt_input_ids[0].tolist()) if tid == self.image_token_id
            ],
            blocked_prompt_positions_after_switch=blocked_prompt_positions,
            token_records=token_records[:len(trimmed_ids)],
        )

        return chat_result, trace


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL generation with image blocked from attention after the first k generated tokens"
    )

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")

    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="JSONL manifest, one example per line.",
    )
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--image-url", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    parser.add_argument("--download-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs_json")

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--k-with-image",
        type=int,
        default=16,
        help="Number of generated tokens that are allowed full direct access to image tokens before masking starts.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument(
        "--also-block-vision-boundary-tokens",
        action="store_true",
        help="Also block vision start/end sentinel tokens after the switch, in addition to the image token span.",
    )

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
        k_with_image=args.k_with_image,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    analyzer = QwenVLMaskedContinuationAnalyzer(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    total_saved = 0
    print("\n=== RUN: Sequential masked continuation generation ===", flush=True)

    for ex_idx, ex in enumerate(resolved_examples, start=1):
        print(
            f"\n[Example {ex_idx}/{len(resolved_examples)}] id={ex.example_id}",
            flush=True,
        )
        print(
            f"Generating with first k={gen_cfg.k_with_image} GENERATED tokens image-conditioned, "
            f"then masking image prompt positions...",
            flush=True,
        )

        chat_result, masked_trace = analyzer.generate_with_image_then_masked_continuation(
            example=ex,
            generation_config=gen_cfg,
            also_block_vision_boundary_tokens=args.also_block_vision_boundary_tokens,
        )

        save_path = save_result_json(
            example=ex,
            chat_result=chat_result,
            masked_trace=masked_trace,
            output_dir=output_dir,
        )
        total_saved += 1

        print(f"  Saved: {save_path}", flush=True)
        print(f"  Generated tokens: {len(chat_result.output_token_ids)}", flush=True)
        print(f"  Answer preview: {chat_result.answer[:200]!r}", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Saved {total_saved} JSON file(s) to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
