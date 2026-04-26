from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, cast

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset import load_track_a_subtask1_eng
from model import BertVARegressor

# (example_id, text, aspect) — id is the same document ID as in JSONL
InferenceItem = Tuple[str, str, str]

_REPO = Path(__file__).resolve().parent
ALL_EVAL_LANGS = ("eng", "jpn", "rus", "tat", "ukr", "zho")


def checkpoint_path(name: str) -> Path:
    return _REPO / "checkpoints" / f"{name}.pth"


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device_arg}', but CUDA is not available on this machine."
            )
        return torch.device(device_arg)

    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "Requested device 'mps', but MPS is not available on this machine."
            )
        return torch.device("mps")

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device '{device_arg}'. Use one of: auto, cpu, cuda, cuda:N, mps."
    )


def make_inference_collate_fn(
    tokenizer: AutoTokenizer,
    max_length: int,
    lang: str,
) -> Callable[[List[InferenceItem]], dict]:
    def collate_fn(batch: List[InferenceItem]) -> dict:
        ex_ids, texts, aspects = zip(*batch)
        formatted_inputs = [f"[CLS] {t} [SEP] {a} [SEP]" for t, a in zip(texts, aspects)]
        tokenized = tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        out: dict = dict(tokenized)
        out["ids"] = list(ex_ids)
        out["lang"] = [lang] * len(batch)
        return out

    return collate_fn


class InferenceVADataset(Dataset[InferenceItem]):
    def __init__(self, items: Sequence[InferenceItem], *, lang: str) -> None:
        self.items = list(items)
        self.lang = lang

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> InferenceItem:
        return self.items[idx]


def _format_va(preds: torch.Tensor) -> str:
    v, a = preds[0].item(), preds[1].item()
    return f"{v:.2f}#{a:.2f}"


def _dataset_lang_dir(lang: str) -> Path:
    return _REPO.parent / "task-dataset" / "track_a" / "subtask_1" / lang


def _prediction_output_path(name: str, lang: str) -> Path:
    return _dataset_lang_dir(lang) / f"{name}_pred.jsonl"


def _results_output_path() -> Path:
    return _REPO / "results.json"


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Pearson correlation inputs must have the same length.")
    if not xs:
        return 0.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def maybe_scale_predictions(preds: torch.Tensor, *, scale_predictions: bool) -> torch.Tensor:
    if not scale_predictions:
        return preds
    return 4.0 * torch.tanh(preds) + 5.0


def _run_test_inference_for_lang(
    model: BertVARegressor,
    device: torch.device,
    tokenizer: AutoTokenizer,
    name: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
    lang: str,
    scale_predictions: bool,
) -> dict[str, float]:
    test_rows = cast(
        List[Tuple[str, str, str, Tuple[float, float], str]],
        load_track_a_subtask1_eng("test", lang=lang),
    )
    flat: List[InferenceItem] = [(doc_id, text, aspect) for doc_id, text, aspect, _va, _src in test_rows]
    gold_vas: List[Tuple[float, float]] = [va for _doc_id, _text, _aspect, va, _src in test_rows]

    if not flat:
        print(f"test[{lang}]: no (text, aspect) pairs; skipping write.")
        return {"RMSE": 0.0, "corrV": 0.0, "corrA": 0.0}

    collate_inf = make_inference_collate_fn(tokenizer, max_length, lang)
    inf_loader = DataLoader(
        InferenceVADataset(flat, lang=lang),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_inf,
    )

    model.eval()
    pred_strings: List[str] = []
    pred_pairs: List[Tuple[float, float]] = []
    with torch.no_grad():
        for batch in tqdm(inf_loader, desc=f"test[{lang}]", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = maybe_scale_predictions(preds, scale_predictions=scale_predictions)
            for row in range(preds.size(0)):
                row_cpu = preds[row].detach().cpu()
                pred_strings.append(_format_va(row_cpu))
                pred_pairs.append((row_cpu[0].item(), row_cpu[1].item()))

    if len(pred_pairs) != len(gold_vas):
        raise RuntimeError("Internal error: prediction count mismatch with gold labels.")

    sq_err_sum = 0.0
    pred_v: List[float] = []
    pred_a: List[float] = []
    gold_v: List[float] = []
    gold_a: List[float] = []
    for (pv, pa), (gv, ga) in zip(pred_pairs, gold_vas):
        sq_err_sum += (pv - gv) ** 2 + (pa - ga) ** 2
        pred_v.append(pv)
        pred_a.append(pa)
        gold_v.append(gv)
        gold_a.append(ga)
    rmse = math.sqrt(sq_err_sum / (2.0 * len(gold_vas)))
    corr_v = _pearson_corr(pred_v, gold_v)
    corr_a = _pearson_corr(pred_a, gold_a)

    aspect_va_by_doc: dict[str, list[dict[str, str]]] = {}
    for row_idx, (doc_id, _text, aspect, _va, _src) in enumerate(test_rows):
        aspect_va_by_doc.setdefault(doc_id, []).append(
            {
                "Aspect": aspect,
                "VA": pred_strings[row_idx],
            }
        )

    out_path = _prediction_output_path(name, lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as w:
        doc_order: List[str] = []
        seen: set[str] = set()
        for doc_id, _text, _aspect, _va, _src in test_rows:
            if doc_id not in seen:
                seen.add(doc_id)
                doc_order.append(doc_id)
        for doc_id in doc_order:
            w.write(
                json.dumps(
                    {
                        "ID": doc_id,
                        "Aspect_VA": aspect_va_by_doc[doc_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote predictions: {out_path} ({len(doc_order)} lines)")
    print(f"RMSE[{lang}]: {rmse:.6f}")
    print(f"corrV[{lang}]: {corr_v:.6f}")
    print(f"corrA[{lang}]: {corr_a:.6f}")
    return {"RMSE": rmse, "corrV": corr_v, "corrA": corr_a}


def _append_results(run_name: str, lang_metrics: dict[str, dict[str, float]]) -> None:
    out_path = _results_output_path()
    if out_path.is_file():
        with out_path.open("r", encoding="utf-8") as r:
            raw = r.read().strip()
        if not raw:
            results_data: dict[str, dict[str, dict[str, float]]] = {}
        else:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                results_data = loaded
            else:
                raise ValueError(
                    f"Expected JSON object in {out_path}, got: {type(loaded)!r}"
                )
    else:
        results_data = {}

    results_data[run_name] = lang_metrics
    with out_path.open("w", encoding="utf-8", newline="\n") as w:
        json.dump(results_data, w, ensure_ascii=False, indent=2)
        w.write("\n")
    print(f"Updated results file: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-set inference from a saved checkpoint.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Run name (must match the checkpoint saved by train: checkpoints/{name}.pth).",
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:N, or mps",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help=(
            "Language subfolder for test mode (e.g. eng), or 'all' to evaluate all supported test languages."
        ),
    )
    parser.add_argument(
        "--scale_predictions",
        action="store_true",
        help="Scale raw model outputs from (-inf, +inf) to [1, 9] using 4*tanh(x)+5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt = checkpoint_path(args.name)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertVARegressor(model_name=args.model_name).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    eval_langs = ALL_EVAL_LANGS if args.lang == "all" else (args.lang,)
    lang_metrics: dict[str, dict[str, float]] = {}
    for eval_lang in eval_langs:
        lang_metrics[eval_lang] = _run_test_inference_for_lang(
            model=model,
            device=device,
            tokenizer=tokenizer,
            name=args.name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            lang=eval_lang,
            scale_predictions=args.scale_predictions,
        )
    _append_results(args.name, lang_metrics)


if __name__ == "__main__":
    main()
