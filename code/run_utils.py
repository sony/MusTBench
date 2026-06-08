"""Execution-loop helpers for Qwen2.5-Omni MusTBench inference."""

import argparse
import os
import traceback
from typing import Any, Dict, List, Sequence, Tuple

from data_utils import (
    batched,
    load_existing_results,
    normalize_track_id,
    reference_answers,
    resolve_audio_path,
    result_key,
    save_json,
)
from model_utils import build_conversation, generate_texts, prepare_transformers_inputs
from parsing import normalize_timepoint, process_generated_output, update_task_metrics
from prompts import MUSTBENCH_TASKS, TASK_CONFIGS


def build_output_row(
    row: Dict[str, Any],
    track_id: str,
    question_text: str,
    model_id: str,
) -> Dict[str, Any]:
    output_row = dict(row)
    output_row["source_type"] = "mustbench"
    output_row["track_id"] = track_id
    output_row["question"] = question_text
    output_row["model_id"] = model_id
    output_row["reference_answers"] = reference_answers(row)
    return output_row

def append_error_result(
    results: List[Dict[str, Any]],
    row: Dict[str, Any],
    message: str,
) -> None:
    row["generated_answer"] = None
    row["generated_text"] = message
    row["model_prediction"] = None
    row["model_prediction_text"] = message
    row["parse_error"] = message
    results.append(row)

def prepare_pending_rows(
    rows: Sequence[Dict[str, Any]],
    output_path: str,
    audio_root: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    existing_results, processed_keys, retry_count = load_existing_results(
        output_path,
        retry_errors=bool(args.retry_errors),
    )

    pending: List[Dict[str, Any]] = []
    skipped_missing_audio = 0
    skipped_invalid_rows = 0
    for row in rows:
        track_id = normalize_track_id(row.get("track_id") or "")
        question_text = str(row.get("question") or "").strip()
        task = str(row.get("task") or "").upper()
        if not track_id or not question_text or task not in TASK_CONFIGS:
            skipped_invalid_rows += 1
            continue
        if result_key(row) in processed_keys:
            continue

        audio_path = resolve_audio_path(track_id, audio_root)
        if args.existing_audio_only and not os.path.exists(audio_path):
            skipped_missing_audio += 1
            continue
        pending.append(row)
    return existing_results, pending, retry_count, skipped_missing_audio, skipped_invalid_rows

def set_default_max_new_tokens(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> None:
    if args.max_new_tokens is not None:
        return
    tasks = {
        str(row.get("task") or "").upper()
        for row in rows
        if str(row.get("task") or "").upper() in TASK_CONFIGS
    }
    if not tasks:
        tasks = set(MUSTBENCH_TASKS)
    args.max_new_tokens = max(int(TASK_CONFIGS[task]["default_max_new_tokens"]) for task in tasks)

def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples is not None and int(args.num_samples) < 1:
        raise ValueError("--num-samples must be >= 1")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_new_tokens is not None and int(args.max_new_tokens) < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if not (0.0 <= float(args.temperature)):
        raise ValueError("--temperature must be >= 0")
    if not (0.0 < float(args.top_p) <= 1.0):
        raise ValueError("--top-p must be in the range (0, 1]")

def process_dataset(
    rows: Sequence[Dict[str, Any]],
    output_path: str,
    audio_root: str,
    model: Any,
    processor: Any,
    torch_module: Any,
    model_id: str,
    args: argparse.Namespace,
) -> None:
    existing_results, pending, retry_count, skipped_missing_audio, skipped_invalid_rows = (
        prepare_pending_rows(rows, output_path, audio_root, args)
    )
    results = list(existing_results)

    print(
        f"[*] mustbench: total={len(rows)}, already_done={len(existing_results)}, "
        f"retry_errors={retry_count}, pending={len(pending)}, "
        f"skipped_missing_audio={skipped_missing_audio}, "
        f"skipped_invalid_rows={skipped_invalid_rows}, "
        f"max_new_tokens={args.max_new_tokens}"
    )

    newly_written = 0
    since_save = 0

    for chunk_index, chunk in enumerate(batched(pending, int(args.batch_size)), start=1):
        conversations = []
        output_rows = []

        for row in chunk:
            track_id = normalize_track_id(row.get("track_id") or "")
            question_text = str(row.get("question") or "").strip()
            task = str(row.get("task") or "").upper()
            output_row = build_output_row(
                row=row,
                track_id=track_id,
                question_text=question_text,
                model_id=model_id,
            )

            audio_path = resolve_audio_path(track_id, audio_root)
            output_row["audio_path"] = audio_path
            if not os.path.exists(audio_path):
                append_error_result(results, output_row, "Error: Audio file not found")
                newly_written += 1
                since_save += 1
                continue

            conversations.append(build_conversation(task, question_text, audio_path))
            output_rows.append(output_row)

        if conversations:
            try:
                inputs = prepare_transformers_inputs(processor, conversations, model, torch_module)
                generated = generate_texts(model, processor, torch_module, inputs, args)
                for output_row, (generated_text, token_count) in zip(output_rows, generated):
                    task = str(output_row.get("task") or "").upper()
                    process_generated_output(task, generated_text, output_row)
                    update_task_metrics(output_row)
                    if token_count is not None:
                        output_row["output_tokens"] = token_count
                    results.append(output_row)
                    newly_written += 1
                    since_save += 1
            except Exception as exc:
                error_text = (
                    f"Error during generation: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                print(error_text)
                for output_row in output_rows:
                    append_error_result(results, output_row, error_text)
                    newly_written += 1
                    since_save += 1
                save_json(output_path, results)
                if not args.continue_on_generation_error:
                    raise RuntimeError(
                        "mustbench generation failed; partial results saved to "
                        f"{output_path}. Fix the root error above, then rerun."
                    ) from exc

        if args.save_every > 0 and since_save >= int(args.save_every):
            save_json(output_path, results)
            print(
                f"[*] mustbench: saved checkpoint after chunk {chunk_index} "
                f"(new={newly_written}/{len(pending)})"
            )
            since_save = 0

    save_json(output_path, results)
    print_summary(output_path, results, newly_written)

def print_preflight_summary(
    rows: Sequence[Dict[str, Any]],
    pending: Sequence[Dict[str, Any]],
    existing_results: Sequence[Dict[str, Any]],
    retry_count: int,
    skipped_missing_audio: int,
    skipped_invalid_rows: int,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    task_counts: Dict[str, int] = {}
    for row in rows:
        task = str(row.get("task") or "").upper()
        task_counts[task] = task_counts.get(task, 0) + 1
    task_text = ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
    print(f"[*] benchmark={args.benchmark_path}")
    print(f"[*] audio_root={args.audio_root}")
    print(f"[*] output_path={output_path}")
    print(f"[*] model_id={args.model_id}")
    print(f"[*] selected_rows={len(rows)} ({task_text})")
    print(
        f"[*] preflight: already_done={len(existing_results)}, retry_errors={retry_count}, "
        f"pending={len(pending)}, skipped_missing_audio={skipped_missing_audio}, "
        f"skipped_invalid_rows={skipped_invalid_rows}"
    )

def print_summary(
    output_path: str,
    results: Sequence[Dict[str, Any]],
    newly_written: int,
) -> None:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        task = str(row.get("task") or "UNKNOWN").upper()
        by_task.setdefault(task, []).append(row)

    print(f"[*] mustbench done. saved={output_path}, new={newly_written}, rows={len(results)}")
    for task in MUSTBENCH_TASKS:
        rows = by_task.get(task, [])
        if not rows:
            continue
        answered = [row for row in rows if str(row.get("generated_answer") or "").strip()]
        parse_errors = [row for row in rows if row.get("parse_error")]
        if task in {"LTR", "GTO"}:
            scored = [row for row in answered if "is_correct" in row]
            correct = [row for row in scored if row.get("is_correct")]
            acc = (len(correct) / len(scored) * 100.0) if scored else 0.0
            print(
                f"[*] {task}: rows={len(rows)}, answered={len(answered)}, "
                f"correct={len(correct)}/{len(scored)}, acc={acc:.2f}%, "
                f"parse_errors={len(parse_errors)}"
            )
        elif task == "TSG":
            scored = [row for row in answered if "is_exact_match" in row]
            exact = [row for row in scored if row.get("is_exact_match")]
            exact_pct = (len(exact) / len(scored) * 100.0) if scored else 0.0
            print(
                f"[*] {task}: rows={len(rows)}, answered={len(answered)}, "
                f"exact={len(exact)}/{len(scored)}, exact={exact_pct:.2f}%, "
                f"parse_errors={len(parse_errors)}"
            )
        else:
            print(
                f"[*] {task}: rows={len(rows)}, answered={len(answered)}, "
                f"parse_errors={len(parse_errors)}"
            )
