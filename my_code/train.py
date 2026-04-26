from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dataset import load_track_a_subtask1_eng
from model import BertVARegressor

# (example_id, text, aspect, (v, a), source_lang) — source_lang is the JSONL subfolder (eng, jpn, …)
Example = Tuple[str, str, str, Tuple[float, float], str]

_REPO = Path(__file__).resolve().parent


def checkpoint_path(name: str) -> Path:
    return _REPO / "checkpoints" / f"{name}.pth"


class VADataset(Dataset[Example]):
    def __init__(self, examples: Sequence[Example], *, lang: str) -> None:
        self.examples = list(examples)
        self.lang = lang

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate_fn(
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Callable[[List[Example]], dict]:
    def collate_fn(batch: List[Example]) -> dict:
        ex_ids, texts, aspects, vas, _src_lang = zip(*batch)
        formatted_inputs = [f"[CLS] {t} [SEP] {a} [SEP]" for t, a in zip(texts, aspects)]
        labels = torch.tensor(list(vas), dtype=torch.float32)

        tokenized = tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        out: dict = dict(tokenized)
        out["labels"] = labels
        out["ids"] = list(ex_ids)
        out["lang"] = list(_src_lang)
        return out

    return collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT regressor for VA prediction.")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Run name: same subdir as test, file {NAME}_<pred_basename> where the test filename has "
            "its first 'test' replaced with 'pred' (e.g. test_eng_laptop.jsonl → {NAME}_pred_eng_laptop.jsonl)"
        ),
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
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
            "Language subfolder, or 'all' → train/dev: eng,jpn,rus,tat,ukr,zho (use test.py for test-mode evaluation)"
        ),
    )
    parser.add_argument(
        "--scale_predictions",
        action="store_true",
        help="Scale raw model outputs from (-inf, +inf) to [1, 9] using 4*tanh(x)+5.",
    )
    return parser.parse_args()


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


def maybe_scale_predictions(preds: torch.Tensor, *, scale_predictions: bool) -> torch.Tensor:
    if not scale_predictions:
        return preds
    return 4.0 * torch.tanh(preds) + 5.0


def train() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"lang={args.lang}")

    train_examples = cast(
        List[Example], load_track_a_subtask1_eng("train", lang=args.lang)
    )
    dev_examples = cast(
        List[Example], load_track_a_subtask1_eng("dev", lang=args.lang)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = VADataset(train_examples, lang=args.lang)
    dev_dataset = VADataset(dev_examples, lang=args.lang)
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = BertVARegressor(model_name=args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(progress, start=1):
            # `batch["ids"]`: one JSONL document ID per row (same for multiple aspects per text).
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = maybe_scale_predictions(preds, scale_predictions=args.scale_predictions)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_running_loss / max(len(train_loader), 1)

        model.eval()
        eval_running_loss = 0.0
        eval_progress = tqdm(dev_loader, desc=f"Eval {epoch}/{args.epochs}", leave=False)
        with torch.no_grad():
            for batch in eval_progress:
                # `batch["ids"]`: document ID per row
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch["labels"].to(device)

                preds = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                preds = maybe_scale_predictions(preds, scale_predictions=args.scale_predictions)
                eval_loss = criterion(preds, labels)
                eval_running_loss += eval_loss.item()
                eval_progress.set_postfix(loss=f"{eval_loss.item():.4f}")

        avg_eval_loss = eval_running_loss / max(len(dev_loader), 1)
        print(
            f"Epoch {epoch}/{args.epochs} - train avg loss: {avg_train_loss:.4f} | "
            f"eval avg loss: {avg_eval_loss:.4f}"
        )

    out_ckpt = checkpoint_path(args.name)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"Saved checkpoint: {out_ckpt}")


if __name__ == "__main__":
    train()
