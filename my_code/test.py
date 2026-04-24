from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, cast

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset import (
    FinalEvalDoc,
    final_eval_prediction_path,
    load_track_a_subtask1_eng,
    resolve_final_eval_langs,
)
from model import BertVARegressor

# (example_id, text, aspect) — id is the same document ID as in JSONL
InferenceItem = Tuple[str, str, str]

_REPO = Path(__file__).resolve().parent


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


def _run_final_eval_inference(
    model: BertVARegressor,
    device: torch.device,
    tokenizer: AutoTokenizer,
    name: str,
    batch_size: int,
    num_workers: int,
    max_length: int,
    lang: str,
) -> None:
    final_docs: List[FinalEvalDoc] = cast(
        List[FinalEvalDoc], load_track_a_subtask1_eng("final_eval", lang=lang)
    )
    flat: List[InferenceItem] = []
    for d in final_docs:
        for aspect in d["aspects"]:
            flat.append((d["id"], d["text"], aspect))

    if not flat:
        print("final_eval: no (text, aspect) pairs; skipping write.")
        return

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
    with torch.no_grad():
        for batch in tqdm(inf_loader, desc="final_eval (test set)", leave=True):
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
            for row in range(preds.size(0)):
                pred_strings.append(_format_va(preds[row].detach().cpu()))

    aspect_va_by_doc: dict[str, list[dict[str, str]]] = {}
    idx = 0
    for d in final_docs:
        part: list[dict[str, str]] = []
        for _asp in d["aspects"]:
            part.append(
                {
                    "Aspect": _asp,
                    "VA": pred_strings[idx],
                }
            )
            idx += 1
        aspect_va_by_doc[d["id"]] = part
    if idx != len(pred_strings):
        raise RuntimeError("Internal error: prediction count mismatch with aspects.")

    out_path = final_eval_prediction_path(name, lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as w:
        for d in final_docs:
            w.write(
                json.dumps(
                    {
                        "ID": d["id"],
                        "Aspect_VA": aspect_va_by_doc[d["id"]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote final eval predictions: {out_path} ({len(final_docs)} lines)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final eval inference from a saved checkpoint.")
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
            "Language subfolder for final_eval, or 'all' → deu, eng, zho (same as train's --lang)"
        ),
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

    for fe_lang in resolve_final_eval_langs(args.lang):
        _run_final_eval_inference(
            model=model,
            device=device,
            tokenizer=tokenizer,
            name=args.name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            lang=fe_lang,
        )


if __name__ == "__main__":
    main()
