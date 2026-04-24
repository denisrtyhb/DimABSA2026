from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple, TypedDict, Union

Mode = Literal["train", "dev", "test", "final_eval"]

# `lang="all"`: which language subfolders to use (order matters for deterministic concatenation)
LANG_ALL_TRAIN_DEV = ("eng", "jpn", "rus", "tat", "ukr", "zho")
LANG_ALL_FINAL_EVAL = ("deu", "eng", "zho")

_MODE_TO_GLOB: Dict[str, str] = {
    "train": "*_train_alltasks.jsonl",
    "dev": "*_dev_task1.jsonl",
    "test": "*_test_task1.jsonl",
}

def _final_eval_subdir_parts(lang: str) -> Tuple[str, ...]:
    return ("evaluation_script", "sample_data", "subtask_1", lang)


class FinalEvalDoc(TypedDict):
    id: str
    text: str
    aspects: List[str]


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_train_dev_langs(lang: str) -> Tuple[str, ...]:
    if lang == "all":
        return LANG_ALL_TRAIN_DEV
    return (lang,)


def resolve_final_eval_langs(lang: str) -> Tuple[str, ...]:
    if lang == "all":
        return LANG_ALL_FINAL_EVAL
    return (lang,)


def final_eval_data_dir(lang: str) -> Path:
    return _repo_root().joinpath(*_final_eval_subdir_parts(lang))


def find_final_eval_test_path(lang: str) -> Path:
    d = final_eval_data_dir(lang)
    if not d.is_dir():
        raise FileNotFoundError(
            f"final_eval directory does not exist: {d!s}"
        )
    matches = sorted(d.glob("test*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"No test*.jsonl in {d!s}. Add exactly one file whose name starts with 'test'."
        )
    if len(matches) > 1:
        names = [p.name for p in matches]
        raise FileNotFoundError(
            f"Expected exactly one test*.jsonl in {d!s}, found {len(matches)}: {names!r}."
        )
    return matches[0]


def final_eval_prediction_path(run_name: str, lang: str) -> Path:
    test_path = find_final_eval_test_path(lang)
    pred_basename = test_path.name.replace("test", "pred", 1)
    return test_path.parent / f"{run_name}_{pred_basename}"


def _parse_line_final_eval(data: dict) -> FinalEvalDoc:
    if "ID" not in data or "Text" not in data:
        raise ValueError("Each final_eval line must have 'ID' and 'Text'.")
    if "Aspect" in data and isinstance(data["Aspect"], list):
        aspects = [a for a in data["Aspect"] if isinstance(a, str)]
    elif "Aspect_VA" in data and isinstance(data["Aspect_VA"], list):
        aspects = [item["Aspect"] for item in data["Aspect_VA"] if isinstance(item, dict) and "Aspect" in item]
    else:
        raise ValueError("Each final_eval line must have 'Aspect' (list) or 'Aspect_VA' (list of dicts).")
    if not aspects:
        raise ValueError(f"No aspects for ID={data.get('ID', '?')!r}.")
    return {"id": str(data["ID"]), "text": str(data["Text"]), "aspects": aspects}


def load_final_eval_subtask1_for_lang(lang: str) -> List[FinalEvalDoc]:
    path = find_final_eval_test_path(lang)

    docs: List[FinalEvalDoc] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                docs.append(_parse_line_final_eval(data))
            except Exception as e:
                raise ValueError(f"{path!s} line {line_num}: {e}") from e
    return docs


# Backwards-compatible alias
def load_final_eval_subtask1_eng(lang: str) -> List[FinalEvalDoc]:
    return load_final_eval_subtask1_for_lang(lang)


def load_track_a_subtask1_eng(
    mode: Mode,
    lang: str = "eng",
) -> Union[
    List[Tuple[str, str, str, Tuple[float, float], str]],
    List[FinalEvalDoc],
]:
    if mode == "final_eval":
        if lang == "all":
            raise ValueError(
                "load_track_a_subtask1_eng(..., 'final_eval', lang='all') is not supported. "
                "Load each language with resolve_final_eval_langs and load_final_eval_subtask1_for_lang, "
                "or use train script which does this automatically."
            )
        return load_final_eval_subtask1_for_lang(lang)

    langs = resolve_train_dev_langs(lang)
    if mode not in _MODE_TO_GLOB:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected one of: train, dev, test, final_eval."
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
    parser.add_argument("mode", choices=["train", "dev", "test", "final_eval"])
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help=(
            "Subfolder (e.g. eng) or 'all' → train/dev: %s; final_eval: %s"
            % (", ".join(LANG_ALL_TRAIN_DEV), ", ".join(LANG_ALL_FINAL_EVAL))
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "final_eval" and args.lang == "all":
        for l in resolve_final_eval_langs("all"):
            p = find_final_eval_test_path(l)
            n = len(load_final_eval_subtask1_for_lang(l))
            print(f"final_eval [{l}] {p!s} -> {n} doc lines")
    else:
        if args.mode == "final_eval":
            print(f"final_eval test file: {find_final_eval_test_path(args.lang)}")
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
