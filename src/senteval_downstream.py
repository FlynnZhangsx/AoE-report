#!/usr/bin/env python
"""Run the seven downstream classification tasks with the official SentEval engine."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# SentEval's 2018 utility calls inspect.getargspec(), which was removed from
# the supported Python 3.10 API for optimizers with keyword-only arguments.
# Keep the official SentEval code unchanged and provide its four-field legacy
# return value at the entry point instead.
def _legacy_getargspec(func):
    spec = inspect.getfullargspec(func)
    return spec.args, spec.varargs, spec.varkw, spec.defaults


inspect.getargspec = _legacy_getargspec


ROOT = Path(__file__).resolve().parents[1]
SENTEVAL_ROOT = ROOT / "third_party" / "SentEval"
sys.path.insert(0, str(SENTEVAL_ROOT))
from senteval.engine import SE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "angle-bert-base-uncased-nli-en-v1")
    ap.add_argument("--task-path", type=Path, default=ROOT / "data" / "senteval")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "downstream" / "official_senteval.json")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--pooling", choices=["cls_avg", "cls", "mean"], default="cls_avg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)
    model.eval()

    @torch.inference_mode()
    def batcher(params, batch):
        if not batch:
            return np.empty((0, int(model.config.hidden_size)), dtype=np.float32)
        texts = [" ".join(tokens) for tokens in batch]
        tok = tokenizer(texts, padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        hidden = model(**tok, return_dict=True).last_hidden_state
        if args.pooling == "cls":
            emb = hidden[:, 0]
        else:
            mask = tok["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            avg = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            emb = (hidden[:, 0] + avg) / 2.0 if args.pooling == "cls_avg" else avg
        return emb.float().cpu().numpy()

    def prepare(params, samples):
        return None

    params = {
        "task_path": str(args.task_path),
        "usepytorch": True,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "kfold": 10,
        "classifier": {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        },
    }
    tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
    logging.info("device=%s gpu=%s model=%s pooling=%s", device,
                 torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                 args.model, args.pooling)
    engine = SE(params, batcher, prepare)
    results = {}
    for task in tasks:
        start = time.time()
        logging.info("starting %s", task)
        results[task] = engine.eval(task)
        results[task]["seconds"] = time.time() - start
        logging.info("finished %s: %s", task, results[task])
    accs = [float(results[x]["acc"]) for x in tasks]
    summary = {
        "model": str(args.model),
        "pooling": args.pooling,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "sent_eval_tasks": tasks,
        "results": results,
        "average_accuracy": float(np.mean(accs)),
        "protocol": "official SentEval engine, default PyTorch logistic classifier, 10-fold where applicable",
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
