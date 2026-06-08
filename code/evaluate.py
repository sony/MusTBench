#!/usr/bin/env python3
"""Evaluate MusTBench inference results for all five benchmark tasks."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from data_utils import normalize_track_id, resolve_audio_path


TASK_TO_TYPE = {
    "TSG": "type1",
    "LTR": "type2",
    "TAD": "type3",
    "GTO": "type4",
    "MTR": "type5",
}
DEFAULT_CLAP_MODEL = "laion/clap-htsat-fused"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TIMEPOINT_RE = re.compile(r"(?<!\d)(?:(\d+):)?(\d+):([0-5]?\d(?:\.\d+)?)(?!\d)")
QUESTION_TIMESTAMP_RE = re.compile(
    r"\bat\s+((?:\d+:)?\d+:[0-5]?\d(?:\.\d+)?)\b",
    re.IGNORECASE,
)
INTERVAL_RE = re.compile(
    r"\b(?:from\s+|between\s+)?"
    r"(\d+(?::\d{1,2}){0,2}(?:\.\d+)?)"
    r"\s*(?:[-~]|to|and)\s*"
    r"(\d+(?::\d{1,2}){0,2}(?:\.\d+)?)\b",
    re.IGNORECASE,
)

Interval = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate inference.py prediction JSONs with Type1 Hit@3s, "
            "Type2/4 accuracy, Type3 METEOR and CLAPScore, and Type5 macro "
            "temporal IoU/F1."
        )
    )
    parser.add_argument("prediction_json", type=Path)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("audios"),
        help="Audio root used to resolve Type3 clips (default: audios).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults next to the prediction JSON.",
    )
    parser.add_argument(
        "--rows-output-path",
        type=Path,
        default=None,
        help="Per-example evaluation JSON path. Defaults next to the prediction JSON.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_TO_TYPE),
        default=None,
        help="Optional task subset.",
    )
    parser.add_argument("--hit-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--clap-model", default=DEFAULT_CLAP_MODEL)
    parser.add_argument(
        "--clap-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--clap-batch-size", type=int, default=32)
    parser.add_argument("--clap-window-seconds", type=float, default=10.0)
    parser.add_argument("--clap-sampling-rate", type=int, default=48000)
    parser.add_argument(
        "--clap-local-files-only",
        action="store_true",
        help="Do not download CLAP model files.",
    )
    return parser.parse_args()


def default_output_paths(prediction_json: Path) -> Tuple[Path, Path]:
    stem = prediction_json.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return (
        prediction_json.with_name(f"{stem}_evaluation.json"),
        prediction_json.with_name(f"{stem}_evaluation_rows.json"),
    )


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(
                f"Expected an object at index {index}, got {type(row).__name__}"
            )
        rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "<na>"}:
        return ""
    return re.sub(r"\s+", " ", text)


def is_error_text(value: Any) -> bool:
    text = normalize_text(value).lower()
    return not text or text.startswith("error")


def as_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return list(converted) if isinstance(converted, list) else [converted]
    return [value]


def reference_values(row: Dict[str, Any]) -> List[str]:
    for key in ("reference_answers", "answers"):
        values = [
            normalize_text(value)
            for value in as_values(row.get(key))
            if normalize_text(value)
        ]
        if values:
            return values
    answer = normalize_text(row.get("answer"))
    return [answer] if answer else []


def prediction_text(row: Dict[str, Any]) -> str:
    for key in (
        "generated_answer",
        "model_prediction",
        "generated_text",
        "model_prediction_text",
    ):
        value = normalize_text(row.get(key))
        if value and not value.lower().startswith("error"):
            return value
    return ""


def parse_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number >= 0.0 else None
    text = normalize_text(value)
    if not text:
        return None
    match = TIMEPOINT_RE.search(text)
    if not match:
        return None
    hours = float(match.group(1) or 0.0)
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_question_timestamp(question: Any) -> Optional[float]:
    match = QUESTION_TIMESTAMP_RE.search(normalize_text(question))
    return parse_timestamp(match.group(1)) if match else None


def parse_choice(value: Any, allowed: str) -> Optional[str]:
    text = normalize_text(value).upper()
    if not text:
        return None
    if text in set(allowed):
        return text
    match = re.search(
        rf"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*([{allowed}])\b",
        text,
    )
    return match.group(1) if match else None


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": mean(numbers),
        "median": median(numbers),
        "std": pstdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
    }


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def meteor_exact(reference: str, prediction: str) -> float:
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_positions: Dict[str, List[int]] = {}
    for index, token in enumerate(ref_tokens):
        ref_positions.setdefault(token, []).append(index)

    used_ref_positions = set()
    matches: List[Tuple[int, int]] = []
    for pred_index, token in enumerate(pred_tokens):
        for ref_index in ref_positions.get(token, []):
            if ref_index not in used_ref_positions:
                used_ref_positions.add(ref_index)
                matches.append((pred_index, ref_index))
                break

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(pred_tokens)
    recall = match_count / len(ref_tokens)
    weighted_f = (10.0 * precision * recall) / (recall + 9.0 * precision)

    matches.sort()
    chunks = 1
    for previous, current in zip(matches, matches[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    penalty = 0.5 * ((chunks / match_count) ** 3)
    return (1.0 - penalty) * weighted_f


def evaluate_type1(
    rows: Sequence[Dict[str, Any]],
    tolerance: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    errors: List[float] = []
    hit_count = 0
    for row in rows:
        references = [
            seconds
            for seconds in (parse_timestamp(value) for value in reference_values(row))
            if seconds is not None
        ]
        prediction = parse_timestamp(prediction_text(row))
        error = (
            min(abs(prediction - reference) for reference in references)
            if prediction is not None and references
            else None
        )
        hit = bool(error is not None and error <= tolerance)
        hit_count += int(hit)
        if error is not None:
            errors.append(error)
        details.append(
            {
                "id": row.get("id"),
                "task": "TSG",
                "type": "type1",
                "track_id": normalize_track_id(row.get("track_id") or ""),
                "reference_seconds": references,
                "prediction_seconds": prediction,
                "absolute_error_seconds": error,
                "hit_at_3s": hit,
            }
        )

    summary = {
        "task": "TSG",
        "metric": "Hit@3s",
        "total": len(rows),
        "parsed_predictions": sum(
            detail["prediction_seconds"] is not None for detail in details
        ),
        "hit_count": hit_count,
        "hit_at_3s": hit_count / len(rows) if rows else None,
        "tolerance_seconds": tolerance,
        "mean_absolute_error_seconds": mean(errors) if errors else None,
        "median_absolute_error_seconds": median(errors) if errors else None,
    }
    return summary, details


def evaluate_choice_rows(
    rows: Sequence[Dict[str, Any]],
    task: str,
    type_name: str,
    allowed: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    correct = 0
    for row in rows:
        references = reference_values(row)
        reference = parse_choice(references[0], allowed) if references else None
        prediction = parse_choice(prediction_text(row), allowed)
        is_correct = bool(
            reference is not None
            and prediction is not None
            and prediction == reference
        )
        correct += int(is_correct)
        details.append(
            {
                "id": row.get("id"),
                "task": task,
                "type": type_name,
                "track_id": normalize_track_id(row.get("track_id") or ""),
                "reference": reference,
                "prediction": prediction,
                "correct": is_correct,
            }
        )

    return (
        {
            "task": task,
            "metric": "Accuracy",
            "total": len(rows),
            "parsed_predictions": sum(
                detail["prediction"] is not None for detail in details
            ),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
        },
        details,
    )


def evaluate_type3_meteor(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    for row in rows:
        references = reference_values(row)
        prediction = prediction_text(row)
        score = (
            max(meteor_exact(reference, prediction) for reference in references)
            if references and prediction
            else 0.0
        )
        details.append(
            {
                "id": row.get("id"),
                "task": "TAD",
                "type": "type3",
                "track_id": normalize_track_id(row.get("track_id") or ""),
                "references": references,
                "prediction": prediction,
                "meteor": score,
                "clapscore": None,
                "clap_audio_path": None,
                "clap_clip_start_seconds": None,
                "clap_clip_end_seconds": None,
            }
        )
    return (
        {
            "task": "TAD",
            "metrics": {
                "meteor": summarize(detail["meteor"] for detail in details),
                "clapscore": None,
            },
            "meteor_backend": "exact-token METEOR without stemming or synonyms",
            "total": len(rows),
        },
        details,
    )


def extract_feature_tensor(output: Any) -> Any:
    return output.pooler_output if hasattr(output, "pooler_output") else output


class ClapAudioTextEmbedder:
    def __init__(
        self,
        model_name: str,
        requested_device: str,
        local_files_only: bool,
        sampling_rate: int,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel

        self.torch = torch
        self.device = (
            "cuda"
            if requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested_device == "auto"
            else requested_device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--clap-device cuda was requested but CUDA is unavailable")
        self.sampling_rate = sampling_rate
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.feature_extractor = ClapFeatureExtractor(sampling_rate=sampling_rate)
        self.model = ClapModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> Any:
        import torch.nn.functional as functional

        batches = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with self.torch.no_grad():
                features = extract_feature_tensor(
                    self.model.get_text_features(**encoded)
                )
                batches.append(
                    functional.normalize(features, p=2, dim=-1).detach().cpu()
                )
        return self.torch.cat(batches, dim=0)

    def encode_audio(
        self,
        audio_path: Path,
        start_seconds: float,
        end_seconds: float,
    ) -> Any:
        import librosa
        import numpy as np
        import torch.nn.functional as functional

        duration = max(0.0, end_seconds - start_seconds)
        audio, sample_rate = librosa.load(
            str(audio_path),
            sr=self.sampling_rate,
            mono=True,
            offset=start_seconds,
            duration=duration,
        )
        target_samples = int(round(duration * sample_rate))
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        encoded = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_features", "is_longer", "attention_mask"}
        }
        with self.torch.no_grad():
            features = extract_feature_tensor(
                self.model.get_audio_features(**encoded)
            )
            return functional.normalize(features, p=2, dim=-1).detach().cpu()[0]


def resolve_row_audio(
    row: Dict[str, Any],
    audio_root: Path,
    prediction_json: Path,
) -> Optional[Path]:
    raw_path = normalize_text(row.get("audio_path"))
    candidates: List[Path] = []
    if raw_path:
        path = Path(raw_path).expanduser()
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(prediction_json.parent / path)
    track_id = normalize_track_id(row.get("track_id") or "")
    if track_id:
        candidates.append(Path(resolve_audio_path(track_id, str(audio_root))))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def add_type3_clap_scores(
    rows: Sequence[Dict[str, Any]],
    details: List[Dict[str, Any]],
    args: argparse.Namespace,
    prediction_json: Path,
) -> Dict[str, Any]:
    valid_indices: List[int] = []
    clip_keys: List[Tuple[Path, float, float]] = []
    missing_audio: List[str] = []

    for index, (row, detail) in enumerate(zip(rows, details)):
        prediction = str(detail["prediction"])
        if is_error_text(prediction):
            detail["clapscore"] = 0.0
            continue
        transition = parse_question_timestamp(row.get("question"))
        if transition is None:
            raise ValueError(
                f"Could not parse TAD timestamp from question for {row.get('id')}: "
                f"{row.get('question')!r}"
            )
        audio_path = resolve_row_audio(
            row,
            args.audio_root.expanduser(),
            prediction_json,
        )
        if audio_path is None:
            missing_audio.append(
                f"{row.get('id')} ({normalize_track_id(row.get('track_id') or '')})"
            )
            continue
        start_seconds = max(0.0, transition - args.clap_window_seconds)
        end_seconds = transition + args.clap_window_seconds
        detail["clap_audio_path"] = str(audio_path)
        detail["clap_clip_start_seconds"] = start_seconds
        detail["clap_clip_end_seconds"] = end_seconds
        valid_indices.append(index)
        clip_keys.append((audio_path, start_seconds, end_seconds))

    if missing_audio:
        preview = ", ".join(missing_audio[:20])
        raise FileNotFoundError(
            f"Could not resolve audio for {len(missing_audio)} TAD rows: {preview}"
        )
    if not valid_indices:
        return {
            "count": len(details),
            "mean": 0.0 if details else None,
            "median": 0.0 if details else None,
            "std": 0.0 if details else None,
            "min": 0.0 if details else None,
            "max": 0.0 if details else None,
        }

    try:
        embedder = ClapAudioTextEmbedder(
            model_name=args.clap_model,
            requested_device=args.clap_device,
            local_files_only=bool(args.clap_local_files_only),
            sampling_rate=int(args.clap_sampling_rate),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CLAP. Install compatible torch, transformers, librosa, "
            "and soundfile packages, or rerun with --skip-clap."
        ) from exc

    text_embeddings = embedder.encode_texts(
        [str(details[index]["prediction"]) for index in valid_indices],
        batch_size=int(args.clap_batch_size),
    )
    audio_cache: Dict[Tuple[Path, float, float], Any] = {}
    for clip_key in sorted(set(clip_keys), key=lambda item: tuple(map(str, item))):
        audio_cache[clip_key] = embedder.encode_audio(*clip_key)

    for index, clip_key, text_embedding in zip(
        valid_indices,
        clip_keys,
        text_embeddings,
    ):
        score = float((audio_cache[clip_key] * text_embedding).sum().item())
        details[index]["clapscore"] = score

    return summarize(
        0.0 if detail["clapscore"] is None else detail["clapscore"]
        for detail in details
    )


def parse_interval_timestamp(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return (
            float(parts[0]) * 3600.0
            + float(parts[1]) * 60.0
            + float(parts[2])
        )
    raise ValueError(f"Unsupported timestamp: {value!r}")


def parse_intervals(value: Any) -> List[Interval]:
    intervals: List[Interval] = []
    for start_text, end_text in INTERVAL_RE.findall(normalize_text(value)):
        try:
            start = parse_interval_timestamp(start_text)
            end = parse_interval_timestamp(end_text)
        except ValueError:
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            intervals.append((start, end))
    return intervals


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    clean = sorted(
        (float(start), float(end))
        for start, end in intervals
        if math.isfinite(start) and math.isfinite(end) and end > start
    )
    if not clean:
        return []
    merged = [clean[0]]
    for start, end in clean[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Sequence[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def intersection_duration(
    left: Sequence[Interval],
    right: Sequence[Interval],
) -> float:
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        total += max(
            0.0,
            min(left_end, right_end) - max(left_start, right_start),
        )
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def temporal_metrics(
    reference_intervals: Sequence[Interval],
    predicted_intervals: Sequence[Interval],
) -> Dict[str, float]:
    reference = merge_intervals(reference_intervals)
    prediction = merge_intervals(predicted_intervals)
    reference_duration = interval_duration(reference)
    prediction_duration = interval_duration(prediction)
    intersection = intersection_duration(reference, prediction)
    union = reference_duration + prediction_duration - intersection
    precision = intersection / prediction_duration if prediction_duration else 0.0
    recall = intersection / reference_duration if reference_duration else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "temporal_iou": intersection / union if union else 0.0,
        "temporal_precision": precision,
        "temporal_recall": recall,
        "temporal_f1": f1,
        "intersection_duration_seconds": intersection,
        "reference_duration_seconds": reference_duration,
        "prediction_duration_seconds": prediction_duration,
    }


def evaluate_type5(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    for row in rows:
        references: List[Interval] = []
        for value in reference_values(row):
            references.extend(parse_intervals(value))

        predictions: List[Interval] = []
        generated_intervals = row.get("generated_intervals")
        if isinstance(generated_intervals, list):
            for value in generated_intervals:
                predictions.extend(parse_intervals(value))
        if not predictions:
            predictions = parse_intervals(prediction_text(row))

        reference_merged = merge_intervals(references)
        prediction_merged = merge_intervals(predictions)
        metrics = temporal_metrics(reference_merged, prediction_merged)
        details.append(
            {
                "id": row.get("id"),
                "task": "MTR",
                "type": "type5",
                "track_id": normalize_track_id(row.get("track_id") or ""),
                "reference_intervals_seconds": reference_merged,
                "prediction_intervals_seconds": prediction_merged,
                **metrics,
            }
        )

    return (
        {
            "task": "MTR",
            "total": len(rows),
            "parsed_predictions": sum(
                bool(detail["prediction_intervals_seconds"])
                for detail in details
            ),
            "macro_temporal_iou": (
                mean(detail["temporal_iou"] for detail in details)
                if details
                else None
            ),
            "macro_temporal_f1": (
                mean(detail["temporal_f1"] for detail in details)
                if details
                else None
            ),
            "macro_temporal_precision": (
                mean(detail["temporal_precision"] for detail in details)
                if details
                else None
            ),
            "macro_temporal_recall": (
                mean(detail["temporal_recall"] for detail in details)
                if details
                else None
            ),
            "interval_definition": "[start, end), duration=end-start",
        },
        details,
    )


def print_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    print(f"[*] evaluated rows: {summary['total_rows']}")
    if "type1" in metrics:
        item = metrics["type1"]
        print(
            f"[*] Type1 / TSG Hit@3s: {item['hit_at_3s']:.6f} "
            f"({item['hit_count']}/{item['total']})"
        )
    if "type2" in metrics:
        item = metrics["type2"]
        print(
            f"[*] Type2 / LTR Accuracy: {item['accuracy']:.6f} "
            f"({item['correct']}/{item['total']})"
        )
    if "type3" in metrics:
        item = metrics["type3"]
        meteor = item["metrics"]["meteor"]["mean"]
        clapscore = item["metrics"]["clapscore"]
        print(f"[*] Type3 / TAD METEOR: {meteor:.6f}")
        print(
            "[*] Type3 / TAD CLAPScore: "
            + ("skipped" if clapscore is None else f"{clapscore['mean']:.6f}")
        )
    if "type4" in metrics:
        item = metrics["type4"]
        print(
            f"[*] Type4 / GTO Accuracy: {item['accuracy']:.6f} "
            f"({item['correct']}/{item['total']})"
        )
    if "type5" in metrics:
        item = metrics["type5"]
        print(f"[*] Type5 / MTR macro Temporal IoU: {item['macro_temporal_iou']:.6f}")
        print(f"[*] Type5 / MTR macro Temporal F1: {item['macro_temporal_f1']:.6f}")
    print(f"[*] summary: {summary['output_path']}")
    print(f"[*] rows: {summary['rows_output_path']}")


def validate_args(args: argparse.Namespace) -> None:
    if args.hit_tolerance_seconds < 0.0:
        raise ValueError("--hit-tolerance-seconds must be >= 0")
    if args.clap_window_seconds <= 0.0:
        raise ValueError("--clap-window-seconds must be > 0")
    if args.clap_batch_size < 1:
        raise ValueError("--clap-batch-size must be >= 1")
    if args.clap_sampling_rate < 1:
        raise ValueError("--clap-sampling-rate must be >= 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    prediction_json = args.prediction_json.expanduser().resolve()
    if not prediction_json.is_file():
        raise FileNotFoundError(prediction_json)

    default_summary, default_rows = default_output_paths(prediction_json)
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else default_summary
    )
    rows_output_path = (
        args.rows_output_path.expanduser().resolve()
        if args.rows_output_path
        else default_rows
    )

    input_rows = load_rows(prediction_json)
    selected_tasks = set(args.tasks or TASK_TO_TYPE)
    rows_by_task = {
        task: [
            row
            for row in input_rows
            if normalize_text(row.get("task")).upper() == task
            and task in selected_tasks
        ]
        for task in TASK_TO_TYPE
    }
    unsupported = sorted(
        {
            normalize_text(row.get("task")).upper()
            for row in input_rows
            if normalize_text(row.get("task")).upper()
            and normalize_text(row.get("task")).upper() not in TASK_TO_TYPE
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported task values: {', '.join(unsupported)}")

    metrics: Dict[str, Any] = {}
    detail_rows: List[Dict[str, Any]] = []

    if rows_by_task["TSG"]:
        metrics["type1"], details = evaluate_type1(
            rows_by_task["TSG"],
            float(args.hit_tolerance_seconds),
        )
        detail_rows.extend(details)
    if rows_by_task["LTR"]:
        metrics["type2"], details = evaluate_choice_rows(
            rows_by_task["LTR"],
            "LTR",
            "type2",
            "ABCD",
        )
        detail_rows.extend(details)
    if rows_by_task["TAD"]:
        metrics["type3"], details = evaluate_type3_meteor(rows_by_task["TAD"])
        if not args.skip_clap:
            metrics["type3"]["metrics"]["clapscore"] = add_type3_clap_scores(
                rows_by_task["TAD"],
                details,
                args,
                prediction_json,
            )
        metrics["type3"]["clap"] = {
            "skipped": bool(args.skip_clap),
            "model": None if args.skip_clap else args.clap_model,
            "device": None if args.skip_clap else args.clap_device,
            "window": (
                None
                if args.skip_clap
                else f"T-{args.clap_window_seconds:g}s to T+{args.clap_window_seconds:g}s"
            ),
        }
        detail_rows.extend(details)
    if rows_by_task["GTO"]:
        metrics["type4"], details = evaluate_choice_rows(
            rows_by_task["GTO"],
            "GTO",
            "type4",
            "ABCDEF",
        )
        detail_rows.extend(details)
    if rows_by_task["MTR"]:
        metrics["type5"], details = evaluate_type5(rows_by_task["MTR"])
        detail_rows.extend(details)

    evaluated_count = sum(len(rows_by_task[task]) for task in selected_tasks)
    summary = {
        "prediction_json": str(prediction_json),
        "output_path": str(output_path),
        "rows_output_path": str(rows_output_path),
        "total_input_rows": len(input_rows),
        "total_rows": evaluated_count,
        "task_mapping": TASK_TO_TYPE,
        "selected_tasks": sorted(selected_tasks),
        "task_counts": {
            task: len(rows_by_task[task])
            for task in TASK_TO_TYPE
            if task in selected_tasks
        },
        "metrics": metrics,
    }
    write_json(rows_output_path, detail_rows)
    write_json(output_path, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
