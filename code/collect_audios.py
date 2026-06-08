#!/usr/bin/env python3
"""Collect the MTG-Jamendo and Slakh2100 audio files used by MusTBench."""

import argparse
import filecmp
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "mustbench.parquet"
SLAKH_SPLITS = ("train", "validation", "test", "omitted")
TRACK_ID_RE = re.compile(r"^Track\d{5}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the unique audio tracks referenced by benchmarks/mustbench.parquet "
            "from downloaded MTG-Jamendo and Slakh2100 datasets."
        )
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help=f"MusTBench parquet path (default: {DEFAULT_BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--mtg-root-path",
        type=Path,
        required=True,
        help=(
            "Absolute path to the MTG-Jamendo audio root or dataset root. "
            "Both <root>/<last_two_digits>/<track_id>.mp3 and "
            "<root>/audios/<last_two_digits>/<track_id>.mp3 are supported."
        ),
    )
    parser.add_argument(
        "--slakh-root-path",
        type=Path,
        required=True,
        help=(
            "Absolute path to the Slakh2100 root, slakh2100_flac_redux root, "
            "or one split directory. Expected files are "
            "<split>/<TrackID>/mix.flac."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory. Audio files are flattened into this directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files that differ from their source.",
    )
    parser.add_argument(
        "--allow-extra-audio",
        action="store_true",
        help="Allow unrelated .mp3/.flac files already present in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate all sources without creating or copying files.",
    )
    return parser.parse_args()


def normalize_track_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_track_ids(benchmark_path: Path) -> Tuple[List[str], List[str]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas and pyarrow are required to read mustbench.parquet."
        ) from exc

    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark parquet not found: {benchmark_path}")

    frame = pd.read_parquet(benchmark_path, columns=["track_id"])
    track_ids = {
        normalize_track_id(value)
        for value in frame["track_id"].dropna().tolist()
        if normalize_track_id(value)
    }

    mtg_ids = sorted(track_id for track_id in track_ids if track_id.isdigit())
    slakh_ids = sorted(track_id for track_id in track_ids if TRACK_ID_RE.fullmatch(track_id))
    unsupported = sorted(track_ids.difference(mtg_ids).difference(slakh_ids))
    if unsupported:
        preview = ", ".join(unsupported[:10])
        raise RuntimeError(f"Unsupported track_id format ({len(unsupported)}): {preview}")

    return mtg_ids, slakh_ids


def unique_existing_files(paths: Iterable[Path]) -> List[Path]:
    found: Dict[Path, Path] = {}
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        found.setdefault(resolved, path)
    return sorted(found.values(), key=lambda item: str(item))


def mtg_audio_roots(root: Path) -> List[Path]:
    candidates = [
        root,
        root / "audios",
        root / "audio",
        root / "raw_30s",
        root / "raw_30s" / "audio",
        root / "raw_30s_audio",
        root / "mtg-jamendo-dataset" / "audios",
    ]
    roots: Dict[Path, Path] = {}
    for candidate in candidates:
        if candidate.is_dir():
            roots.setdefault(candidate.resolve(), candidate)
    return list(roots.values())


def direct_mtg_candidates(track_id: str, roots: Sequence[Path]) -> List[Path]:
    bucket = track_id[-2:].zfill(2)
    filename = f"{track_id}.mp3"
    return [
        path
        for root in roots
        for path in (root / bucket / filename, root / filename)
    ]


def index_missing_mtg_tracks(root: Path, missing_ids: Set[str]) -> Dict[str, List[Path]]:
    wanted_names = {f"{track_id}.mp3": track_id for track_id in missing_ids}
    found: Dict[str, List[Path]] = {track_id: [] for track_id in missing_ids}
    if not wanted_names:
        return found

    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            track_id = wanted_names.get(filename)
            if track_id is not None:
                found[track_id].append(Path(directory) / filename)
    return found


def resolve_mtg_tracks(root: Path, track_ids: Sequence[str]) -> Dict[str, Path]:
    roots = mtg_audio_roots(root)
    resolved: Dict[str, Path] = {}
    missing: Set[str] = set()

    for track_id in track_ids:
        matches = unique_existing_files(direct_mtg_candidates(track_id, roots))
        if len(matches) == 1:
            resolved[track_id] = matches[0]
        elif len(matches) > 1:
            paths = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(f"Multiple MTG sources found for {track_id}:\n  {paths}")
        else:
            missing.add(track_id)

    fallback = index_missing_mtg_tracks(root, missing)
    for track_id in sorted(missing):
        matches = unique_existing_files(fallback[track_id])
        if len(matches) == 1:
            resolved[track_id] = matches[0]
        elif len(matches) > 1:
            paths = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(f"Multiple MTG sources found for {track_id}:\n  {paths}")

    return resolved


def slakh_dataset_roots(root: Path) -> List[Path]:
    candidates = [
        root,
        root / "slakh2100_flac_redux",
        root / "slakh2100" / "slakh2100_flac_redux",
    ]
    roots: Dict[Path, Path] = {}
    for candidate in candidates:
        if candidate.is_dir():
            roots.setdefault(candidate.resolve(), candidate)
    return list(roots.values())


def direct_slakh_candidates(track_id: str, roots: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = []
    for root in roots:
        candidates.append(root / track_id / "mix.flac")
        candidates.extend(root / split / track_id / "mix.flac" for split in SLAKH_SPLITS)
        candidates.append(root / f"{track_id}.flac")
    return candidates


def index_missing_slakh_tracks(root: Path, missing_ids: Set[str]) -> Dict[str, List[Path]]:
    found: Dict[str, List[Path]] = {track_id: [] for track_id in missing_ids}
    if not missing_ids:
        return found

    for directory, subdirs, filenames in os.walk(root):
        directory_path = Path(directory)
        name = directory_path.name
        if TRACK_ID_RE.fullmatch(name):
            if name in missing_ids and "mix.flac" in filenames:
                found[name].append(directory_path / "mix.flac")
            subdirs[:] = []
            continue

        for filename in filenames:
            if filename.endswith(".flac") and filename[:-5] in missing_ids:
                found[filename[:-5]].append(directory_path / filename)
    return found


def resolve_slakh_tracks(root: Path, track_ids: Sequence[str]) -> Dict[str, Path]:
    roots = slakh_dataset_roots(root)
    resolved: Dict[str, Path] = {}
    missing: Set[str] = set()

    for track_id in track_ids:
        matches = unique_existing_files(direct_slakh_candidates(track_id, roots))
        if len(matches) == 1:
            resolved[track_id] = matches[0]
        elif len(matches) > 1:
            paths = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(f"Multiple Slakh sources found for {track_id}:\n  {paths}")
        else:
            missing.add(track_id)

    fallback = index_missing_slakh_tracks(root, missing)
    for track_id in sorted(missing):
        matches = unique_existing_files(fallback[track_id])
        if len(matches) == 1:
            resolved[track_id] = matches[0]
        elif len(matches) > 1:
            paths = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(f"Multiple Slakh sources found for {track_id}:\n  {paths}")

    return resolved


def destination_name(track_id: str, source: Path) -> str:
    return f"{track_id}{source.suffix.lower()}"


def validate_resolved_tracks(
    mtg_ids: Sequence[str],
    slakh_ids: Sequence[str],
    mtg_sources: Dict[str, Path],
    slakh_sources: Dict[str, Path],
) -> None:
    missing_mtg = sorted(set(mtg_ids).difference(mtg_sources))
    missing_slakh = sorted(set(slakh_ids).difference(slakh_sources))
    if not missing_mtg and not missing_slakh:
        return

    lines = ["Could not resolve every MusTBench audio source."]
    if missing_mtg:
        lines.append(
            f"Missing MTG-Jamendo tracks ({len(missing_mtg)}): "
            + ", ".join(missing_mtg[:20])
        )
    if missing_slakh:
        lines.append(
            f"Missing Slakh2100 tracks ({len(missing_slakh)}): "
            + ", ".join(missing_slakh[:20])
        )
    raise RuntimeError("\n".join(lines))


def validate_output_dir(
    output_dir: Path,
    expected_names: Set[str],
    allow_extra_audio: bool,
) -> None:
    if not output_dir.exists() or allow_extra_audio:
        return

    extra = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".mp3", ".flac"}
        and path.name not in expected_names
    )
    if extra:
        raise RuntimeError(
            f"Output directory contains unrelated audio files ({len(extra)}): "
            + ", ".join(extra[:20])
        )


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_tracks(
    sources: Dict[str, Path],
    output_dir: Path,
    overwrite: bool,
) -> Tuple[int, int]:
    copied = 0
    unchanged = 0
    for index, (track_id, source) in enumerate(sorted(sources.items()), start=1):
        destination = output_dir / destination_name(track_id, source)
        if destination.exists():
            if filecmp.cmp(source, destination, shallow=False):
                unchanged += 1
                continue
            if not overwrite:
                raise RuntimeError(
                    f"Destination differs from source: {destination}. "
                    "Use --overwrite to replace it."
                )

        copy_atomic(source, destination)
        copied += 1
        if index % 50 == 0 or index == len(sources):
            print(f"[*] copied/resolved {index}/{len(sources)}")
    return copied, unchanged


def require_absolute_directory(path: Path, option_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{option_name} must be an absolute path: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{option_name} is not a directory: {path}")
    return path.resolve()


def main() -> None:
    args = parse_args()
    benchmark_path = args.benchmark_path.expanduser().resolve()
    mtg_root = require_absolute_directory(
        args.mtg_root_path.expanduser(), "--mtg-root-path"
    )
    slakh_root = require_absolute_directory(
        args.slakh_root_path.expanduser(), "--slakh-root-path"
    )
    output_dir = args.output_dir.expanduser().resolve()

    mtg_ids, slakh_ids = load_track_ids(benchmark_path)
    print(
        f"[*] benchmark tracks: total={len(mtg_ids) + len(slakh_ids)}, "
        f"mtg={len(mtg_ids)}, slakh={len(slakh_ids)}"
    )

    mtg_sources = resolve_mtg_tracks(mtg_root, mtg_ids)
    slakh_sources = resolve_slakh_tracks(slakh_root, slakh_ids)
    validate_resolved_tracks(mtg_ids, slakh_ids, mtg_sources, slakh_sources)

    sources = {**mtg_sources, **slakh_sources}
    expected_names = {
        destination_name(track_id, source)
        for track_id, source in sources.items()
    }
    if len(expected_names) != len(sources):
        raise RuntimeError("Multiple tracks map to the same destination filename.")

    validate_output_dir(output_dir, expected_names, args.allow_extra_audio)
    print(
        f"[*] resolved all sources: total={len(sources)}, "
        f"mp3={sum(path.suffix.lower() == '.mp3' for path in sources.values())}, "
        f"flac={sum(path.suffix.lower() == '.flac' for path in sources.values())}"
    )

    if args.dry_run:
        print(f"[*] dry run complete; no files copied to {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    copied, unchanged = copy_tracks(sources, output_dir, args.overwrite)
    validate_output_dir(output_dir, expected_names, args.allow_extra_audio)

    present = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name in expected_names
    }
    missing_outputs = sorted(expected_names.difference(present))
    if missing_outputs:
        raise RuntimeError(
            f"Copy completed with missing outputs ({len(missing_outputs)}): "
            + ", ".join(missing_outputs[:20])
        )

    print(
        f"[*] complete: expected={len(expected_names)}, copied={copied}, "
        f"unchanged={unchanged}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
