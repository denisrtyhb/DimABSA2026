from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

Mode = Literal["train", "dev", "test"]

# `lang="all"`: which language subfolders to use (order matters for deterministic concatenation)
LANG_ALL_TRAIN_DEV = ("eng", "jpn", "rus", "tat", "ukr", "zho")

_MODE_TO_GLOB: Dict[str, str] = {
    "train": "*_train_alltasks.jsonl",
    "dev": "*_dev_task1.jsonl",
    "test": "*_test_task1.jsonl",
}

def _dataset_dir(lang: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "task-dataset"
        / "track_a"
        / "subtask_1"
        / lang
    )

def extract_VA(VA: str) -> Tuple[float, float]:
    return tuple(float(v) for v in VA.split("#"))

def parse_examples(example: dict) -> List[Tuple[str, str, str, Tuple[float, float]]]:
    ex_id = str(example["ID"])
    if "Quadruplet" in example:
        return [
            (ex_id, example["Text"], quad["Aspect"], extract_VA(quad["VA"]))
            for quad in example["Quadruplet"]
        ]
    assert "Aspect_VA" in example, "Aspect_VA not found in example"

    return [
        (ex_id, example["Text"], av["Aspect"], extract_VA(av["VA"]))
        for av in example["Aspect_VA"]
    ]


def resolve_train_dev_langs(lang: str) -> Tuple[str, ...]:
    if lang == "all":
        return LANG_ALL_TRAIN_DEV
    return (lang,)


def load_track_a_subtask1_eng(
    mode: Mode,
    lang: str = "eng",
) -> List[Tuple[str, str, str, Tuple[float, float], str]]:
    langs = resolve_train_dev_langs(lang)
    if mode not in _MODE_TO_GLOB:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: train, dev, test."
        )

    pattern = _MODE_TO_GLOB[mode]
    multi = len(langs) > 1
    rows: List[Tuple[str, str, str, Tuple[float, float], str]] = []

    for sub in langs:
        dataset_path = _dataset_dir(sub)
        files = sorted(dataset_path.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files found for mode '{mode}' in '{dataset_path}' with pattern '{pattern}'."
            )
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cur = json.loads(line)
                        for ex_id, text, aspect, va in parse_examples(cur):
                            ex_id_n = f"{sub}::{ex_id}" if multi else ex_id
                            rows.append((ex_id_n, text, aspect, va, sub))

    return rows



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load track_a/subtask_1/{lang} datasets for train/dev/test mode."
    )
    parser.add_argument("mode", choices=["train", "dev", "test"])
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help=(
            "Subfolder (e.g. eng) or 'all' → train/dev/test: %s"
            % ", ".join(LANG_ALL_TRAIN_DEV)
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_track_a_subtask1_eng(args.mode, lang=args.lang)
    print(f"Loaded {len(data)} rows for mode='{args.mode}' lang={args.lang!r}.")
    if data:
        print(data[0])
    if len(data) > 1:
        print(data[1])
    if len(data) > 2:
        print(data[2])
    if len(data) > 3:
        print(data[3])
