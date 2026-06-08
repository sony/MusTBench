#!/usr/bin/env python3
"""Run Qwen2.5-Omni-7B Transformers inference for MusTBench QA parquet."""

import argparse
import os
import sys

from data_utils import (
    default_benchmark_path,
    default_output_path,
    load_mustbench_rows,
)
from model_utils import (
    load_model_and_processor,
)
from prompts import (
    MUSTBENCH_TASKS,
)
from run_utils import (
    prepare_pending_rows,
    print_preflight_summary,
    process_dataset,
    set_default_max_new_tokens,
    validate_args,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Omni-7B Transformers inference for MusTBench QA parquet."
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=default_benchmark_path(),
        help="Path to mustbench.parquet.",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        required=True,
        help=(
            "Path to the audio root. Supports files under "
            "{audio_root}/{track_id} or {audio_root}/{last_two_digits}/{track_id} "
            "with .mp3, .wav, .flac, .m4a, or .ogg extensions."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Output path. Defaults to "
            "results/<benchmark_stem>_<model_id>_predictions.json."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=MUSTBENCH_TASKS,
        default=None,
        help="Optional task filter. Choices: TSG LTR TAD GTO MTR.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Hugging Face model id to load, e.g. Qwen/Qwen2.5-Omni-7B.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map passed to from_pretrained. Use 'auto' by default.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="Optional attention implementation, e.g. flash_attention_2. Use auto to omit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Rows passed to each Transformers generate call. Use 1 if GPU memory is tight.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. Defaults to the largest needed value for selected tasks.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--existing-audio-only",
        dest="existing_audio_only",
        action="store_true",
        default=True,
        help="Skip rows whose audio file is missing (default).",
    )
    parser.add_argument(
        "--allow-missing-audio",
        dest="existing_audio_only",
        action="store_false",
        help="Write error rows for missing audio instead of skipping them.",
    )
    parser.add_argument(
        "--retry-errors",
        dest="retry_errors",
        action="store_true",
        default=True,
        help="Retry existing rows whose previous output is an error or parse error (default).",
    )
    parser.add_argument(
        "--keep-error-results",
        dest="retry_errors",
        action="store_false",
        help="Treat existing error rows as processed when resuming.",
    )
    parser.add_argument(
        "--continue-on-generation-error",
        action="store_true",
        help="Write errors for a failed batch and continue. Default is fail-fast after saving.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, filters, resume state, and audio availability without loading the model.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    if not os.path.isfile(args.benchmark_path):
        raise FileNotFoundError(f"Benchmark parquet not found: {args.benchmark_path}")
    if not os.path.isdir(args.audio_root):
        raise FileNotFoundError(f"Audio root not found: {args.audio_root}")

    rows = load_mustbench_rows(
        benchmark_path=args.benchmark_path,
        tasks=args.tasks,
        num_samples=args.num_samples,
    )
    set_default_max_new_tokens(args, rows)
    validate_args(args)

    output_path = args.output_path or default_output_path(args.benchmark_path, args.model_id)
    existing_results, pending, retry_count, skipped_missing_audio, skipped_invalid_rows = (
        prepare_pending_rows(rows, output_path, args.audio_root, args)
    )
    print_preflight_summary(
        rows=rows,
        pending=pending,
        existing_results=existing_results,
        retry_count=retry_count,
        skipped_missing_audio=skipped_missing_audio,
        skipped_invalid_rows=skipped_invalid_rows,
        output_path=output_path,
        args=args,
    )
    if args.preflight_only:
        return
    if not pending:
        print("[*] No pending rows to process.")
        return

    model, processor, torch_module = load_model_and_processor(str(args.model_id), args)
    process_dataset(
        rows=rows,
        output_path=output_path,
        audio_root=args.audio_root,
        model=model,
        processor=processor,
        torch_module=torch_module,
        model_id=str(args.model_id),
        args=args,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        sys.exit(130)
