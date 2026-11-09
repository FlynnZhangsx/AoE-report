#!/usr/bin/env python
"""In-domain AoE experiments for the paper's STS-B/GIS setting.

This is a small, dependency-free implementation of the official AnglE
Format-A objective.  It starts from the released BERT-base AoE checkpoint,
uses the model-card ``cls_avg`` pooling, and fine-tunes separately on STS-B
and GIS.  The official AoE setting is in-batch-negative plus 0.02 times the
angle ranking loss; a cosine-only control can also be requested.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "angle-bert-base-uncased-nli-en-v1"
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "in_domain_strict"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[str, str, float]]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate_fn(batch, tokenizer, max_length):
    # AnglE's Format-A collator emits a zigzag sequence:
    # [text1_a, text1_b, text2_a, text2_b, ...].  The ranking losses
    # deliberately index [::2] and [1::2], so preserving this order is
    # essential.
    texts = [text for row in batch for text in row[:2]]
    labels = torch.tensor([float(x[2]) for x in batch], dtype=torch.float32)
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tok, labels


def read_task(data_root: Path, task: str):
    if task == "stsb":
        base = data_root / "stsbenchmark"
        files = {"train": base / "train.parquet", "valid": base / "validation.parquet", "test": base / "test.parquet"}
        cols = ("sentence1", "sentence2", "score")
    elif task == "gis":
        base = data_root / "gis"
        files = {"train": base / "train-00000-of-00001.parquet", "valid": base / "valid-00000-of-00001.parquet", "test": base / "test-00000-of-00001.parquet"}
        cols = ("text1", "text2", "label")
    else:
        raise ValueError(task)
    out = {}
    for split, path in files.items():
        df = pd.read_parquet(path)
        out[split] = [(str(a), str(b), float(y)) for a, b, y in zip(df[cols[0]], df[cols[1]], df[cols[2]])]
    return out


def cosine_loss(labels: torch.Tensor, outputs: torch.Tensor, tau: float = 20.0):
    labels = labels[::2]
    target = (labels[:, None] < labels[None, :]).float()
    scores = F.normalize(outputs, p=2, dim=1)
    scores = (scores[::2] * scores[1::2]).sum(dim=1) * tau
    diff = (scores[:, None] - scores[None, :])
    diff = (diff - (1.0 - target) * 1e12).reshape(-1)
    return torch.logsumexp(torch.cat([torch.zeros(1, device=outputs.device), diff]), dim=0)


def angle_loss(labels: torch.Tensor, outputs: torch.Tensor, tau: float = 20.0):
    labels = labels[::2]
    target = (labels[:, None] < labels[None, :]).float()
    re, im = torch.chunk(outputs, 2, dim=1)
    a, b = re[::2], im[::2]
    c, d = re[1::2], im[1::2]
    z = torch.sum(c.square() + d.square(), dim=1, keepdim=True).clamp_min(1e-8)
    q_re = (a * c + b * d) / z
    q_im = (b * c - a * d) / z
    dz = torch.sum(a.square() + b.square(), dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    dw = torch.sum(c.square() + d.square(), dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    q_re = q_re / (dz / dw)
    q_im = q_im / (dz / dw)
    # Official implementation: pool signed re/im first, then abs.
    scores = torch.abs(torch.cat([q_re, q_im], dim=1).sum(dim=1)) * tau
    diff = (scores[:, None] - scores[None, :])
    diff = (diff - (1.0 - target) * 1e12).reshape(-1)
    return torch.logsumexp(torch.cat([torch.zeros(1, device=outputs.device), diff]), dim=0)


def ibn_loss(labels: torch.Tensor, outputs: torch.Tensor, tau: float = 20.0):
    """Official AnglE Format-A in-batch-negative loss."""
    device = outputs.device

    def make_target(y):
        idx = torch.arange(outputs.shape[0], device=device, dtype=torch.int32)
        y = y.int()
        i1 = idx[None, :].clone()
        i2 = (idx + 1 - idx % 2 * 2)[:, None]
        i1 = i1 * y.T
        i1 = i1 + (y.T == 0).int() * -2
        i2 = i2 * y
        i2 = i2 + (y == 0).int() * -1
        return (i1 == i2).float()

    neg_mask = make_target(labels == 0)
    target = make_target(labels)
    norm = F.normalize(outputs, p=2, dim=1)
    sim = (norm @ norm.T) - torch.eye(outputs.shape[0], device=device) * 1e12
    sim = sim * tau + neg_mask * 0.0
    return (-(F.log_softmax(sim, dim=1) * target).sum(dim=1)).mean()


def pair_embeddings(model, tok, device, pooling="cls_avg"):
    tok = {k: v.to(device) for k, v in tok.items()}
    hidden = model(**tok, return_dict=True).last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    avg = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "cls_avg":
        return (hidden[:, 0] + avg) / 2.0
    if pooling == "mean":
        return avg
    raise ValueError(f"Unsupported pooling={pooling}")


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, batch_size, max_length, pooling):
    model.eval()
    pred, gold = [], []
    loader = DataLoader(PairDataset(rows), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_length))
    for tok, labels in loader:
        emb = pair_embeddings(model, tok, device, pooling)
        a, b = emb[::2], emb[1::2]
        score = (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1)
        pred.extend(score.float().cpu().numpy().tolist())
        gold.extend(labels.numpy().tolist())
    return float(spearmanr(gold, pred).statistic), len(gold)


def train_one(task, objective, rows, valid_rows, test_rows, args):
    seed_all(args.seed)
    run_dir = args.output_dir / task / objective
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device)
    model.train()
    if task == "gis":
        max_length, batch_size = args.gis_max_length, args.gis_batch_size
    else:
        max_length, batch_size = args.stsb_max_length, args.stsb_batch_size
    loader = DataLoader(PairDataset(rows), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
                        num_workers=0, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = GradScaler(enabled=args.fp16 and args.device.type == "cuda")
    best = -math.inf
    best_epoch = 0
    log_file = run_dir / "training.jsonl"
    with log_file.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for tok, labels in loader:
                labels_zig = labels.repeat_interleave(2).to(args.device)
                optim.zero_grad(set_to_none=True)
                with autocast(enabled=args.fp16 and args.device.type == "cuda"):
                    out = pair_embeddings(model, tok, args.device, args.pooling)
                    loss = torch.zeros((), device=args.device)
                    if objective == "cosine":
                        loss = loss + args.cosine_weight * cosine_loss(labels_zig, out, args.cosine_tau)
                    if objective in {"ibn", "aoe"}:
                        loss = loss + args.ibn_weight * ibn_loss(labels_zig[:, None], out, args.ibn_tau)
                    if objective in {"angle", "aoe"}:
                        loss = loss + args.angle_weight * angle_loss(labels_zig, out, args.angle_tau)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            val, nval = evaluate(model, valid_rows, tokenizer, args.device, args.eval_batch_size, max_length, args.pooling)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_spearman": val,
                   "n_train": len(rows), "n_valid": nval, "task": task, "objective": objective,
                   "seed": args.seed, "max_length": max_length}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps({"event": "epoch", **row}), flush=True)
            if val > best:
                best, best_epoch = val, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_spearman": val}, run_dir / "best.pt")
    ckpt = torch.load(run_dir / "best.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test, ntest = evaluate(model, test_rows, tokenizer, args.device, args.eval_batch_size, max_length, args.pooling)
    result = {"task": task, "objective": objective, "best_epoch": best_epoch,
              "validation_spearman": best, "test_spearman": test, "n_test": ntest,
              "model_init": str(args.model), "pooling": args.pooling, "max_length": max_length,
              "learning_rate": args.learning_rate, "epochs": args.epochs,
              "batch_size": batch_size, "seed": args.seed}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", nargs="+", default=["stsb", "gis"])
    p.add_argument("--objectives", nargs="+", default=["aoe"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stsb-batch-size", type=int, default=32)
    p.add_argument("--gis-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--stsb-max-length", type=int, default=128)
    p.add_argument("--gis-max-length", type=int, default=512)
    p.add_argument("--pooling", choices=["cls_avg", "cls", "mean"], default="cls_avg")
    p.add_argument("--cosine-weight", type=float, default=1.0)
    p.add_argument("--ibn-weight", type=float, default=1.0)
    p.add_argument("--angle-weight", type=float, default=0.02)
    p.add_argument("--cosine-tau", type=float, default=20.0)
    p.add_argument("--ibn-tau", type=float, default=20.0)
    p.add_argument("--angle-tau", type=float, default=20.0)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "device": str(args.device),
                      "gpu": torch.cuda.get_device_name(0) if args.device.type == "cuda" else None,
                      "model": str(args.model), "tasks": args.tasks,
                      "objectives": args.objectives}), flush=True)
    all_results = []
    for task in args.tasks:
        splits = read_task(args.data_root, task)
        for objective in args.objectives:
            all_results.append(train_one(task, objective, splits["train"], splits["valid"], splits["test"], args))
    (args.output_dir / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "results": all_results}), flush=True)


if __name__ == "__main__":
    main()
