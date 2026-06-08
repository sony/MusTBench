"""Task prompts and response contracts for MusTBench inference."""

from typing import Any, Dict

MUSTBENCH_TASKS = ("TSG", "LTR", "TAD", "GTO", "MTR")

TSG_SYSTEM_PROMPT = (
    "You are an expert in music analysis.\n"
    "Answer timestamp grounding questions by identifying when the requested musical event occurs.\n"
    "Use only evidence from the audio.\n"
    "Output exactly one line in this format: ANSWER: <TIME>.\n"
    "<TIME> must use numeric M:SS format, for example 0:37 or 2:05.\n"
    "Do not output explanations, markdown, or any additional text."
)

LTR_SYSTEM_PROMPT = (
    "You are an expert in music analysis.\n"
    "Identify the musical change at the asked timestamp and choose the best option.\n"
    "Output exactly one line in this format: ANSWER: <LETTER>.\n"
    "<LETTER> must be one of A, B, C, or D.\n"
    "Do not output explanations, markdown, or any additional text."
)

TAD_SYSTEM_PROMPT = (
    "You are an expert in music analysis.\n"
    "Describe the musical change at the asked timestamp in detail.\n"
    "Use only evidence from the audio.\n"
    "Output exactly one line in this format: ANSWER: <DESCRIPTION>.\n"
    "Do not mention timestamp text in the generated description.\n"
    "Do not output explanations, markdown, or any additional text."
)

GTO_SYSTEM_PROMPT = (
    "You are an expert in music analysis.\n"
    "Three transition descriptions (X, Y, Z) are given; determine their chronological order in the track.\n"
    "Then map the earliest-to-latest order to the provided A-F option list.\n"
    "Output exactly one line in this format: ANSWER: <LETTER>.\n"
    "<LETTER> must be one of A, B, C, D, E, or F.\n"
    "Do not output explanations, transition orders, timestamps, markdown, or any additional text."
)

MTR_SYSTEM_PROMPT = (
    "You are an expert in music arousal analysis.\n"
    "Arousal refers to the perceived activation level of the music: how energetic, intense, active, or calm a section feels.\n"
    "Identify all time interval(s) with the highest or lowest arousal duration(s) asked in the question.\n"
    "Listen to the whole audio and compare sections by their overall perceived activation level.\n"
    "Output exactly one line in this format: ANSWER: <INTERVALS>.\n"
    "Each interval must use numeric M:SS-M:SS format, for example 0:35-1:17.\n"
    "If there are multiple intervals, separate them with commas.\n"
    "Do not output placeholder timestamps, explanations, markdown, or any additional text."
)

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "TSG": {
        "mode": "timepoint",
        "system_prompt": TSG_SYSTEM_PROMPT,
        "allowed_letters": None,
        "default_max_new_tokens": 8,
    },
    "LTR": {
        "mode": "choice",
        "system_prompt": LTR_SYSTEM_PROMPT,
        "allowed_letters": {"A", "B", "C", "D"},
        "default_max_new_tokens": 8,
    },
    "TAD": {
        "mode": "open_ended",
        "system_prompt": TAD_SYSTEM_PROMPT,
        "allowed_letters": None,
        "default_max_new_tokens": 128,
    },
    "GTO": {
        "mode": "choice",
        "system_prompt": GTO_SYSTEM_PROMPT,
        "allowed_letters": {"A", "B", "C", "D", "E", "F"},
        "default_max_new_tokens": 8,
    },
    "MTR": {
        "mode": "timespan",
        "system_prompt": MTR_SYSTEM_PROMPT,
        "allowed_letters": None,
        "default_max_new_tokens": 32,
    },
}
