"""Qwen2.5-Omni Transformers loading and generation helpers."""

import argparse
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prompts import TASK_CONFIGS


def build_user_text(question_text: str) -> str:
    text = str(question_text or "").strip()
    if re.match(r"^\s*Question\s*:", text, re.IGNORECASE):
        return text
    return f"Question:\n{text}"

def build_conversation(task: str, question_text: str, audio_path: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": str(TASK_CONFIGS[task]["system_prompt"])}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": build_user_text(question_text)},
            ],
        },
    ]

def torch_dtype_from_name(name: str, torch_module: Any) -> Any:
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    return "auto"

def get_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device

def get_model_dtype(model: Any) -> Any:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype

def cast_floating_inputs(inputs: Any, torch_module: Any, target_dtype: Any) -> Any:
    for key, value in inputs.items():
        if torch_module.is_tensor(value) and torch_module.is_floating_point(value):
            if target_dtype != "auto" and value.dtype != target_dtype:
                inputs[key] = value.to(dtype=target_dtype)
    return inputs

def load_model_and_processor(model_id: str, args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing Qwen2.5-Omni runtime dependency. Install torch, accelerate, "
            "and a transformers version with Qwen2.5-Omni support."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Your transformers version does not expose Qwen2_5OmniProcessor and "
            "Qwen2_5OmniThinkerForConditionalGeneration. Install a Qwen2.5-Omni-compatible "
            "transformers release."
        ) from exc

    print(f"[*] Loading processor from: {model_id}")
    try:
        processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)
    except TypeError:
        processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    load_kwargs: Dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype_from_name(str(args.torch_dtype), torch),
    }
    if str(args.attn_implementation).lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"[*] Loading Qwen2.5-Omni Thinker model from: {model_id}")
    print(
        f"[*] Model load settings: device_map={args.device_map}, "
        f"torch_dtype={args.torch_dtype}, attn_implementation={args.attn_implementation}"
    )
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **load_kwargs)

    model.eval()
    print("[*] Model loaded successfully.")
    return model, processor, torch

def prepare_transformers_inputs(
    processor: Any,
    conversations: Sequence[List[Dict[str, Any]]],
    model: Any,
    torch_module: Any,
) -> Any:
    payload: Any = conversations[0] if len(conversations) == 1 else list(conversations)
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": False,
    }
    try:
        inputs = processor.apply_chat_template(payload, **kwargs)
    except TypeError as exc:
        if "use_audio_in_video" not in str(exc):
            raise
        kwargs.pop("use_audio_in_video")
        inputs = processor.apply_chat_template(payload, **kwargs)

    inputs = inputs.to(get_model_device(model))
    return cast_floating_inputs(inputs, torch_module, get_model_dtype(model))

def generation_kwargs(args: argparse.Namespace, use_thinker_prefix: bool = False) -> Dict[str, Any]:
    token_arg = "thinker_max_new_tokens" if use_thinker_prefix else "max_new_tokens"
    kwargs: Dict[str, Any] = {
        token_arg: int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
    }
    if args.do_sample:
        kwargs["temperature"] = float(args.temperature)
        kwargs["top_p"] = float(args.top_p)
    return kwargs

def generate_texts(
    model: Any,
    processor: Any,
    torch_module: Any,
    inputs: Any,
    args: argparse.Namespace,
) -> List[Tuple[str, Optional[int]]]:
    base_kwargs = generation_kwargs(args)
    candidates = [
        {**base_kwargs, "use_audio_in_video": False},
        dict(base_kwargs),
    ]

    last_exc: Optional[Exception] = None
    outputs = None
    with torch_module.no_grad():
        for kwargs in candidates:
            try:
                outputs = model.generate(**inputs, **kwargs)
                break
            except TypeError as exc:
                last_exc = exc
        if outputs is None:
            raise last_exc if last_exc is not None else RuntimeError("Generation failed")

    generated_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    if hasattr(generated_ids, "sequences"):
        generated_ids = generated_ids.sequences

    input_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
    if hasattr(generated_ids, "shape") and len(generated_ids.shape) == 2 and generated_ids.shape[1] > input_len:
        decoded_ids = generated_ids[:, input_len:]
    else:
        decoded_ids = generated_ids

    texts = processor.batch_decode(
        decoded_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    token_counts: List[Optional[int]] = []
    if hasattr(decoded_ids, "shape") and len(decoded_ids.shape) == 2:
        token_counts = [int(decoded_ids.shape[1])] * int(decoded_ids.shape[0])
    else:
        token_counts = [None] * len(texts)
    return [(text.strip(), token_count) for text, token_count in zip(texts, token_counts)]
