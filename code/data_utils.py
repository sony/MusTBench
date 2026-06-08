"""Data, path, and audio helpers for MusTBench inference."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def sanitize_model_name(model: str) -> str:
    text = str(model or "").strip()
    if "/" in text:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "unknown_model"

def save_json(path: str, payload: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(list(payload), f, indent=2, ensure_ascii=False)

def normalize_track_id(value: Any) -> str:
    text = str(value).strip()
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = text.replace("\\", "/").rsplit("/", 1)[-1].strip()
    while text.lower().endswith(".mp3"):
        text = text[:-4].rstrip(" .")
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text

def is_retryable_error_row(row: Dict[str, Any]) -> bool:
    generated_answer = row.get("generated_answer")
    generated_text = str(row.get("generated_text", "")).strip()
    prediction_text = str(row.get("model_prediction_text", "")).strip()
    model_prediction = str(row.get("model_prediction", "")).strip()
    generated_answer_text = str(generated_answer or "").strip()
    if row.get("parse_error"):
        return True
    if generated_answer_text.startswith("Error") or model_prediction.startswith("Error"):
        return True
    if generated_text.startswith("Error") or prediction_text.startswith("Error"):
        return True
    if generated_answer is None:
        return generated_text.startswith("Error") or prediction_text.startswith("Error")
    return False

def batched(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]

def default_benchmark_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "benchmarks" / "mustbench.parquet")

def default_output_path(benchmark_path: str, model_id: str) -> str:
    stem = Path(benchmark_path).stem
    label = sanitize_model_name(model_id)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    return str(results_dir / f"{stem}_{label}_predictions.json")

def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}

def json_safe(value: Any) -> Any:
    if is_missing_value(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)

def iter_answer_values(value: Any) -> Iterable[Any]:
    if is_missing_value(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]

def reference_answers(row: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for value in iter_answer_values(row.get("answers")):
        if not is_missing_value(value):
            refs.append(str(value).strip())
    if not refs and not is_missing_value(row.get("answer")):
        refs.append(str(row["answer"]).strip())
    return refs

def load_mustbench_rows(
    benchmark_path: str,
    tasks: Optional[Sequence[str]],
    num_samples: Optional[int],
) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: pandas with parquet support. Install pandas and pyarrow."
        ) from exc

    df = pd.read_parquet(benchmark_path)
    required_columns = {
        "id",
        "track_id",
        "task",
        "category",
        "qa_type",
        "question",
        "answer",
        "answers",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Benchmark parquet is missing required columns: {', '.join(missing_columns)}"
        )

    if tasks:
        task_set = {str(task).upper() for task in tasks}
        df = df[df["task"].astype(str).str.upper().isin(task_set)]
    if num_samples is not None:
        df = df.head(int(num_samples))

    rows: List[Dict[str, Any]] = []
    for row_index, record in df.iterrows():
        row = {column: json_safe(record.get(column)) for column in df.columns}
        row["row_index"] = int(row_index)
        if row.get("task") is not None:
            row["task"] = str(row["task"]).upper()
        rows.append(row)
    return rows

def result_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    row_id = str(row.get("id") or "").strip()
    if row_id:
        return ("id", row_id)
    track_id = normalize_track_id(row.get("track_id") or "")
    question = str(row.get("question") or "").strip()
    return ("qa", track_id, question)

def load_existing_results(
    output_path: str,
    retry_errors: bool,
) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, ...]], int]:
    if not os.path.exists(output_path):
        return [], set(), 0

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise RuntimeError("JSON root is not a list")
    except Exception as exc:
        print(f"[Warning] Existing output is unreadable and will be overwritten: {exc}")
        return [], set(), 0

    kept_rows: List[Dict[str, Any]] = []
    processed: Set[Tuple[str, ...]] = set()
    retry_count = 0
    for row in rows:
        if retry_errors and is_retryable_error_row(row):
            retry_count += 1
            continue
        kept_rows.append(row)
        processed.add(result_key(row))
    return kept_rows, processed, retry_count

def candidate_audio_paths(track_id: str, audio_root: str) -> List[Path]:
    raw = str(track_id).strip()
    normalized = normalize_track_id(raw)
    root = Path(audio_root)
    candidates: List[Path] = []
    seen: Set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            candidates.append(path)
            seen.add(key)

    raw_path = Path(raw)
    if raw and (raw_path.suffix or "/" in raw or "\\" in raw):
        add(raw_path if raw_path.is_absolute() else root / raw_path)

    extensions = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    if normalized:
        suffix = Path(normalized).suffix
        if suffix:
            add(root / normalized)
        else:
            for ext in extensions:
                add(root / f"{normalized}{ext}")

            if normalized.isdigit():
                bucket = normalized[-2:].zfill(2)
                for ext in extensions:
                    add(root / bucket / f"{normalized}{ext}")

    return candidates

def resolve_audio_path(track_id: str, audio_root: str) -> str:
    candidates = candidate_audio_paths(track_id, audio_root)
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0]) if candidates else str(Path(audio_root) / f"{track_id}.mp3")
