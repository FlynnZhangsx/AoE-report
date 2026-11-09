"""Create report figures and subset diagnostics from a completed run."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from aoe_repro import PairDataset, Projection, angle_scores, pair_cosine


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stsbenchmark"
OUT = ROOT / "results"


def predict(run_name: str, pooling: str, objective: str, split: str):
    ds = PairDataset(DATA / f"features/{pooling}_{split}.npz")
    out_dim = 128 if objective == "cosine" else 256
    model = Projection(128, out_dim, complex_init=out_dim == 256)
    model.load_state_dict(torch.load(OUT / run_name / "best.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        outputs = torch.stack([model(ds.x1), model(ds.x2)], dim=1).reshape(-1, out_dim)
        pred = pair_cosine(outputs).numpy()
        angle = angle_scores(outputs).numpy() if out_dim == 256 else np.full(len(ds), np.nan)
    return ds.y.numpy(), pred, angle


def main():
    rows = []
    for run, pooling, objective in [
        ("baseline_cosine_cls", "cls", "cosine"),
        ("aoe_cls", "cls", "aoe"),
        ("ablation_angle_only_cls", "cls", "angle"),
        ("aoe_mean_pooling", "mean", "aoe"),
        ("aoe_cls_seed7", "cls", "aoe"),
    ]:
        y, pred, angle = predict(run, pooling, objective, "test")
        for subset, mask in {
            "all": np.ones(len(y), dtype=bool),
            "high_similarity_(y>=4)": y >= 4,
            "low_similarity_(y<=2)": y <= 2,
        }.items():
            rows.append({
                "run": run,
                "subset": subset,
                "n": int(mask.sum()),
                "spearman": float(spearmanr(y[mask], pred[mask]).statistic)
                if mask.sum() > 2 else np.nan,
                "mean_pred": float(pred[mask].mean()),
                "mean_angle_score": float(angle[mask].mean())
                if not np.isnan(angle).all() else np.nan,
            })
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT / "subset_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in ["baseline_cosine_cls", "aoe_cls", "ablation_angle_only_cls", "aoe_mean_pooling"]:
        log = pd.read_json(OUT / run / "training.jsonl", lines=True)
        ax.plot(log["epoch"], log["val_spearman"], marker="o", label=run)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Spearman")
    ax.set_title("AoE reproduction: validation curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "validation_curves.png", dpi=180)
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
