#!/usr/bin/env python
"""Strict evaluation of the released AoE BERT checkpoint.

This follows the paper's standard STS protocol (Spearman correlation) and
the model-card pooling recipe (``cls_avg``).  It intentionally does not
fine-tune or substitute a smaller backbone: the checkpoint is the released
``SeanLee97/angle-bert-base-uncased-nli-en-v1`` model trained on SNLI+MNLI.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "angle-bert-base-uncased-nli-en-v1"
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "strict"


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def batched(items: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.inference_mode()
def encode(
    texts: Sequence[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for batch in batched(list(texts), batch_size):
        tok = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model(**tok, return_dict=True).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1).to(out.dtype)
        avg = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        # Exact AnglE ``cls_avg`` pooling from the official source:
        # (CLS + attention-mask mean) / 2.
        emb = (out[:, 0] + avg) / 2.0
        chunks.append(emb.float().cpu().numpy())
    if not chunks:
        return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def score_pairs(
    pairs: Sequence[Tuple[str, str, float]], tokenizer, model, device, batch_size, max_length
) -> Tuple[float, np.ndarray, np.ndarray]:
    left = [x[0] for x in pairs]
    right = [x[1] for x in pairs]
    labels = np.asarray([float(x[2]) for x in pairs], dtype=np.float64)
    # Encode both sides in one call to avoid duplicate tokenization and keep
    # ordering deterministic.
    emb = encode(left + right, tokenizer, model, device, batch_size, max_length)
    a, b = emb[: len(left)], emb[len(left) :]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    pred = (a * b).sum(axis=1) / np.maximum(denom, 1e-12)
    corr = float(spearmanr(labels, pred).statistic)
    return corr, labels, pred


def read_jsonl_gz(path: Path) -> List[Tuple[str, str, float]]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                out.append((str(x["sentence1"]), str(x["sentence2"]), float(x["score"])))
    return out


def read_stsb(path: Path) -> List[Tuple[str, str, float]]:
    df = pd.read_parquet(path)
    return [(str(a), str(b), float(y)) for a, b, y in zip(df.sentence1, df.sentence2, df.score)]


def read_gis(path: Path) -> List[Tuple[str, str, float]]:
    df = pd.read_parquet(path)
    return [(str(a), str(b), float(y)) for a, b, y in zip(df.text1, df.text2, df.label)]


def load_tasks(data_root: Path):
    mteb = data_root / "mteb"
    tasks = {}
    for name in ["sts12", "sts13", "sts14", "sts15", "sts16", "sickr"]:
        tasks[name.upper() if name != "sickr" else "SICK-R"] = read_jsonl_gz(
            mteb / f"{name}-sts.jsonl.gz"
        )
    tasks["STS-B"] = read_stsb(data_root / "stsbenchmark" / "test.parquet")
    tasks["GIS"] = read_gis(data_root / "gis" / "test-00000-of-00001.parquet")
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible")
    dev_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if dev_name == "auto":
        dev_name = "cpu"
    device = torch.device(dev_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "event": "start",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "python": platform.python_version(),
        "model": str(args.model),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True)
    model.to(device)
    model.eval()
    tasks = load_tasks(args.data_root)
    if args.tasks:
        wanted = {x.upper() for x in args.tasks}
        tasks = {k: v for k, v in tasks.items() if k.upper() in wanted}

    rows = []
    for name, pairs in tasks.items():
        t0 = time.time()
        corr, labels, pred = score_pairs(
            pairs, tokenizer, model, device, args.batch_size, args.max_length
        )
        row = {"task": name, "n": len(pairs), "spearman": corr, "seconds": time.time() - t0}
        rows.append(row)
        np.savez_compressed(args.output_dir / f"{name.replace('-', '_')}_predictions.npz", labels=labels, predictions=pred)
        print(json.dumps({"event": "task", **row}), flush=True)

    standard = [r["spearman"] for r in rows if r["task"] in {"STS12", "STS13", "STS14", "STS15", "STS16", "SICK-R", "STS-B"}]
    summary = {
        "model": str(args.model),
        "pooling": "cls_avg",
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "tasks": rows,
        "standard_sts_average": float(np.mean(standard)) if standard else None,
        "strict_claim": "released_official_checkpoint_evaluation",
        "model_sha256": sha256(args.model / "pytorch_model.bin"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "n", "spearman", "seconds"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"event": "done", "summary": str(args.output_dir / 'summary.json')}), flush=True)


if __name__ == "__main__":
    main()
