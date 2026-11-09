"""Lightweight AoE reproduction on STSBenchmark.

The frozen BERT-Tiny encoder keeps the CPU experiment practical. The trainable
128->256 projection is split into real and imaginary halves for the AoE angle
objective. Losses follow the public AnglE implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "stsbenchmark"
DEFAULT_BACKBONE = ROOT / "models" / "bert-tiny"
DEFAULT_OUT = ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    return df[["sentence1", "sentence2", "score"]].copy()


def _mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def encode_texts(
    texts: List[str],
    backbone_dir: Path,
    pooling: str = "cls",
    max_length: int = 64,
    batch_size: int = 128,
) -> np.ndarray:
    tokenizer = BertTokenizer.from_pretrained(str(backbone_dir), local_files_only=True)
    model = BertModel.from_pretrained(str(backbone_dir), local_files_only=True)
    model.eval()
    result: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tok = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        out = model(**tok).last_hidden_state
        if pooling == "cls":
            emb = out[:, 0, :]
        elif pooling == "mean":
            emb = _mean_pool(out, tok["attention_mask"])
        else:
            raise ValueError(f"Unsupported pooling={pooling}")
        result.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(result, axis=0)


def build_feature_cache(
    data_dir: Path,
    cache_dir: Path,
    backbone_dir: Path,
    pooling: str,
    max_length: int,
    batch_size: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {s: read_split(data_dir, s) for s in ("train", "validation", "test")}
    texts: List[str] = []
    for df in frames.values():
        texts.extend(df["sentence1"].tolist())
        texts.extend(df["sentence2"].tolist())
    unique = list(dict.fromkeys(texts))
    print(f"[features] pooling={pooling} unique_texts={len(unique)}")
    t0 = time.time()
    emb = encode_texts(unique, backbone_dir, pooling, max_length, batch_size)
    lookup = {text: emb[i] for i, text in enumerate(unique)}
    print(f"[features] encoded shape={emb.shape} seconds={time.time() - t0:.1f}")
    for split, df in frames.items():
        x1 = np.stack([lookup[x] for x in df["sentence1"]])
        x2 = np.stack([lookup[x] for x in df["sentence2"]])
        np.savez_compressed(cache_dir / f"{pooling}_{split}.npz", x1=x1, x2=x2,
                            y=df["score"].to_numpy(np.float32))


class PairDataset(Dataset):
    def __init__(self, npz_path: Path):
        arr = np.load(npz_path)
        self.x1 = torch.from_numpy(arr["x1"])
        self.x2 = torch.from_numpy(arr["x2"])
        self.y = torch.from_numpy(arr["y"])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x1[idx], self.x2[idx], self.y[idx]


class Projection(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 128, complex_init: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        if complex_init and out_dim == 2 * in_dim:
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.weight[:in_dim].copy_(torch.eye(in_dim))
                self.proj.weight[in_dim:].normal_(mean=0.0, std=0.01)
                self.proj.bias.zero_()
        elif out_dim == in_dim:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(in_dim))
                self.proj.bias.zero_()
        else:
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _pair_ranking_loss(labels: torch.Tensor, scores: torch.Tensor, tau: float) -> torch.Tensor:
    """CoSENT-style ranking loss used by the public AnglE code."""
    labels = labels.reshape(-1)
    target = (labels[:, None] < labels[None, :]).float()
    diff = (scores * tau)[:, None] - (scores * tau)[None, :]
    diff = (diff - (1.0 - target) * 1e12).reshape(-1)
    return torch.logsumexp(torch.cat([torch.zeros(1, device=scores.device), diff]), dim=0)


def cosine_loss(labels: torch.Tensor, outputs: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    outputs = F.normalize(outputs, p=2, dim=1)
    pair_scores = (outputs[::2] * outputs[1::2]).sum(dim=1)
    return _pair_ranking_loss(labels[::2], pair_scores, tau)


def angle_scores(outputs: torch.Tensor, pooling: str = "sum") -> torch.Tensor:
    re, im = torch.chunk(outputs, 2, dim=1)
    a, b = re[::2], im[::2]
    c, d = re[1::2], im[1::2]
    denom = (c.square() + d.square()).sum(dim=1, keepdim=True).clamp_min(1e-8)
    quotient_re = (a * c + b * d) / denom
    quotient_im = (b * c - a * d) / denom
    dz = (a.square() + b.square()).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    dw = (c.square() + d.square()).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    quotient_re = quotient_re / (dz / dw)
    quotient_im = quotient_im / (dz / dw)
    normalized = torch.cat([quotient_re, quotient_im], dim=1)
    if pooling == "sum":
        # Official AnglE implementation pools the signed real/imaginary
        # components first, then takes the absolute angular difference.
        pooled = normalized.sum(dim=1)
    elif pooling == "mean":
        pooled = normalized.mean(dim=1)
    else:
        raise ValueError(f"Unsupported angle pooling={pooling}")
    return pooled.abs()


def angle_loss(labels: torch.Tensor, outputs: torch.Tensor, tau: float = 20.0,
               pooling: str = "sum") -> torch.Tensor:
    return _pair_ranking_loss(labels[::2], angle_scores(outputs, pooling), tau)


def pair_cosine(outputs: torch.Tensor) -> torch.Tensor:
    outputs = F.normalize(outputs, p=2, dim=1)
    return (outputs[::2] * outputs[1::2]).sum(dim=1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, mode: str) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []
    for x1, x2, y in loader:
        outputs = torch.stack([model(x1), model(x2)], dim=1).reshape(-1, model.proj.out_features)
        preds.append(pair_cosine(outputs).cpu().numpy())
        ys.append(y.numpy())
    pred_np, y_np = np.concatenate(preds), np.concatenate(ys)
    rho = float(spearmanr(y_np, pred_np).statistic)
    return {"spearman": rho, "n": int(len(y_np)), "mode": mode}


def train_one(
    name: str,
    pooling: str,
    objective: str,
    seed: int,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    angle_tau: float,
    angle_weight: float,
) -> Dict[str, float]:
    set_seed(seed)
    train_ds = PairDataset(data_dir / f"features/{pooling}_train.npz")
    valid_ds = PairDataset(data_dir / f"features/{pooling}_validation.npz")
    test_ds = PairDataset(data_dir / f"features/{pooling}_test.npz")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    out_dim = 128 if objective == "cosine" else 256
    model = Projection(in_dim=train_ds.x1.shape[1], out_dim=out_dim,
                       complex_init=(out_dim == 256))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.jsonl"
    best = {"spearman": -math.inf, "epoch": 0}
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for x1, x2, y in train_loader:
                outputs = torch.stack([model(x1), model(x2)], dim=1).reshape(-1, model.proj.out_features)
                zigzag_labels = y[:, None].repeat_interleave(2, dim=0)
                loss = torch.zeros(())
                if objective in {"cosine", "aoe"}:
                    loss = loss + cosine_loss(zigzag_labels, outputs)
                if objective in {"angle", "aoe"}:
                    loss = loss + angle_weight * angle_loss(
                        zigzag_labels, outputs, tau=angle_tau, pooling="sum")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                losses.append(float(loss.detach()))
            valid = evaluate(model, valid_loader, objective)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_spearman": valid["spearman"],
                "seconds": time.time() - t0,
                "seed": seed,
                "pooling": pooling,
                "objective": objective,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{name}] epoch={epoch:02d} loss={row['train_loss']:.5f} "
                  f"val_spearman={row['val_spearman']:.4f}")
            if valid["spearman"] > best["spearman"]:
                best = {"spearman": valid["spearman"], "epoch": epoch}
                torch.save(model.state_dict(), run_dir / "best.pt")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    test = evaluate(model, test_loader, objective)
    result = {
        "name": name,
        "objective": objective,
        "pooling": pooling,
        "seed": seed,
        "best_epoch": best["epoch"],
        "validation_spearman": best["spearman"],
        "test_spearman": test["spearman"],
        "angle_weight": angle_weight,
        "angle_tau": angle_tau,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_size": len(train_ds),
        "validation_size": len(valid_ds),
        "test_size": len(test_ds),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def ensure_features(args: argparse.Namespace) -> None:
    feature_dir = args.data / "features"
    for pooling in args.poolings:
        expected = [feature_dir / f"{pooling}_{s}.npz"
                    for s in ("train", "validation", "test")]
        if all(x.exists() for x in expected):
            print(f"[features] cache hit pooling={pooling}")
            continue
        build_feature_cache(args.data, feature_dir, args.backbone, pooling,
                            args.max_length, args.feature_batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poolings", nargs="+", default=["cls"])
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--angle-tau", type=float, default=20.0)
    parser.add_argument("--angle-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_features:
        ensure_features(args)
    configs = [
        ("baseline_cosine_cls", "cls", "cosine", args.seed),
        ("aoe_cls", "cls", "aoe", args.seed),
        ("ablation_angle_only_cls", "cls", "angle", args.seed),
    ]
    if "mean" in args.poolings:
        configs.append(("aoe_mean_pooling", "mean", "aoe", args.seed))
    configs.append(("aoe_cls_seed7", "cls", "aoe", 7))
    results = []
    for name, pooling, objective, seed in configs:
        results.append(train_one(
            name=name, pooling=pooling, objective=objective, seed=seed,
            data_dir=args.data, out_dir=args.out, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, angle_tau=args.angle_tau,
            angle_weight=args.angle_weight))
    pd.DataFrame(results).to_csv(args.out / "results.csv", index=False)
    (args.out / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
