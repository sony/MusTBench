"""Model-output parsing and lightweight MusTBench scoring helpers."""

import re
from typing import Any, Dict, List, Optional, Set

from data_utils import reference_answers
from prompts import TASK_CONFIGS

TIMEPOINT_RE = re.compile(r"\b(\d{1,3}):(\d{1,2})\b")
INTERVAL_RE = re.compile(r"\b(\d{1,3}:\d{1,2})\s*[-~]\s*(\d{1,3}:\d{1,2})\b")
ANSWER_RE = re.compile(r"(?im)\bANSWER\b\s*\**\s*:\s*\**\s*(.*?)\s*$")


def extract_answer_payload(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    matches = [m.group(1).strip().strip("*").strip() for m in ANSWER_RE.finditer(stripped)]
    matches = [match for match in matches if match]
    if not matches:
        block_match = re.search(r"(?is)\bANSWER\b\s*\**\s*:\s*\**\s*(.+)", stripped)
        if block_match:
            first_payload_line = block_match.group(1).strip().splitlines()[0]
            first_payload_line = first_payload_line.strip().strip("*").strip()
            if first_payload_line:
                matches = [first_payload_line]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and len(set(matches)) == 1:
        return matches[-1]
    return None

def extract_choice_letter(text: str, allowed_letters: Set[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    alphabet = "".join(sorted(allowed_letters))
    strict_pattern = re.compile(
        rf"(?im)^\s*ANSWER\s*:\s*\**\s*([{alphabet}])\s*\**\s*[.)]?\s*$"
    )
    matches = [m.group(1).upper() for m in strict_pattern.finditer(stripped)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and len(set(matches)) == 1:
        return matches[-1]

    one_line = re.compile(rf"^\s*\**\s*([{alphabet}])\s*\**\s*[.)]?\s*$", re.I)
    match = one_line.match(stripped)
    if match:
        return match.group(1).upper()

    answer_line = re.compile(
        rf"(?im)^\s*(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|-)?\s*\**\s*([{alphabet}])\b"
    )
    matches = [m.group(1).upper() for m in answer_line.finditer(stripped)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and len(set(matches)) == 1:
        return matches[-1]
    return None

def seconds_from_timepoint(time_text: str) -> Optional[int]:
    match = TIMEPOINT_RE.fullmatch(str(time_text).strip())
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    if seconds >= 60:
        return None
    return minutes * 60 + seconds

def normalize_timepoint(time_text: str) -> Optional[str]:
    total_seconds = seconds_from_timepoint(time_text)
    if total_seconds is None:
        return None
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"

def extract_timepoints(text: str) -> List[str]:
    timepoints: List[str] = []
    seen: Set[str] = set()
    for minute_text, second_text in TIMEPOINT_RE.findall(text or ""):
        normalized = normalize_timepoint(f"{minute_text}:{second_text}")
        if normalized and normalized not in seen:
            timepoints.append(normalized)
            seen.add(normalized)
    return timepoints

def extract_time_intervals(text: str) -> List[str]:
    intervals: List[str] = []
    seen: Set[str] = set()
    for start_text, end_text in INTERVAL_RE.findall(text or ""):
        start_seconds = seconds_from_timepoint(start_text)
        end_seconds = seconds_from_timepoint(end_text)
        if start_seconds is None or end_seconds is None or end_seconds <= start_seconds:
            continue
        start = normalize_timepoint(start_text)
        end = normalize_timepoint(end_text)
        if start is None or end is None:
            continue
        interval = f"{start}-{end}"
        if interval not in seen:
            intervals.append(interval)
            seen.add(interval)
    return intervals

def process_generated_output(
    task: str,
    generated_text: str,
    output_row: Dict[str, Any],
) -> None:
    cfg = TASK_CONFIGS[task]
    mode = str(cfg["mode"])
    output_row["generated_text"] = generated_text
    output_row["model_prediction_text"] = generated_text
    output_row.pop("parse_error", None)
    output_row.pop("parse_warning", None)

    if mode == "choice":
        generated_answer = extract_choice_letter(generated_text, cfg["allowed_letters"])
        if generated_answer is None:
            parse_error = "Error: Invalid response format. Expected 'ANSWER: <LETTER>'."
            output_row["generated_answer"] = parse_error
            output_row["model_prediction"] = parse_error
            output_row["parse_error"] = parse_error
        else:
            output_row["generated_answer"] = generated_answer
            output_row["model_prediction"] = generated_answer
        return

    if mode == "timepoint":
        answer_payload = extract_answer_payload(generated_text)
        search_text = answer_payload if answer_payload is not None else generated_text
        timepoints = extract_timepoints(search_text)
        if not timepoints:
            output_row["generated_answer"] = None
            output_row["generated_timepoints"] = []
            output_row["model_prediction"] = None
            output_row["parse_error"] = "Error: Invalid response format. Expected 'ANSWER: M:SS'."
            return
        if answer_payload is None:
            output_row["parse_warning"] = "Missing ANSWER prefix; parsed timestamp from raw response."
        elif len(timepoints) > 1:
            output_row["parse_warning"] = "Multiple timestamps found; used the first timestamp."
        output_row["generated_answer"] = timepoints[0]
        output_row["generated_timepoints"] = timepoints
        output_row["model_prediction"] = timepoints[0]
        return

    if mode == "timespan":
        answer_payload = extract_answer_payload(generated_text)
        search_text = answer_payload if answer_payload is not None else generated_text
        intervals = extract_time_intervals(search_text)
        if not intervals:
            output_row["generated_answer"] = None
            output_row["generated_intervals"] = []
            output_row["model_prediction"] = None
            output_row["parse_error"] = "Error: Invalid response format. Expected 'ANSWER: <INTERVALS>'."
            return
        if answer_payload is None:
            output_row["parse_warning"] = "Missing ANSWER prefix; parsed intervals from raw response."
        generated_answer = ", ".join(intervals)
        output_row["generated_answer"] = generated_answer
        output_row["generated_intervals"] = intervals
        output_row["model_prediction"] = generated_answer
        return

    if mode == "open_ended":
        generated_answer = extract_answer_payload(generated_text)
        if generated_answer is None:
            generated_answer = generated_text.strip()
            if not generated_answer or generated_answer.startswith("Error"):
                output_row["generated_answer"] = None
                output_row["model_prediction"] = None
                output_row["parse_error"] = "Error: Invalid response format. Expected 'ANSWER: <DESCRIPTION>'."
                return
            output_row["parse_warning"] = "Missing ANSWER prefix; used raw response text."
        output_row["generated_answer"] = generated_answer
        output_row["model_prediction"] = generated_answer
        return

    raise ValueError(f"Unsupported mode: {mode}")

def update_task_metrics(output_row: Dict[str, Any]) -> None:
    task = str(output_row.get("task") or "").upper()
    cfg = TASK_CONFIGS.get(task)
    if not cfg:
        return

    mode = str(cfg["mode"])
    references = reference_answers(output_row)
    pred = str(output_row.get("generated_answer") or "").strip()
    if not references or not pred:
        return

    if mode == "choice":
        allowed_letters = cfg["allowed_letters"]
        gold = references[0].strip().upper()
        pred_letter = pred.upper()
        if gold in allowed_letters and pred_letter in allowed_letters:
            output_row["is_correct"] = bool(pred_letter == gold)
    elif mode == "timepoint":
        normalized_refs = [normalize_timepoint(ref) for ref in references]
        normalized_refs = [ref for ref in normalized_refs if ref is not None]
        if normalized_refs:
            output_row["is_exact_match"] = pred in normalized_refs
