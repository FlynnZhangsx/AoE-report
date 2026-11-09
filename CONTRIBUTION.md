# 个人实际工作及理解证据

实验编号和路径可由日志核查；提交前可将“本人”替换为实际姓名。

## 本人

- 负责范围：论文检索、AoE 复合空间角度目标实现、CPU 可复现实验设计。
- 代码产出：`src/aoe_repro.py`（特征缓存、complex division、angle loss、训练/评测）。
- 分析产出：`src/analyze_results.py`、`results/subset_metrics.csv`、
  `results/validation_curves.png`。
- 实验产出：`results/console.log`、五组 `results/*/training.jsonl`、
  `results/results.csv`。
- 环境/数据产出：`ENVIRONMENT.md`、`data/stsbenchmark/`、`models/bert-tiny/`。
- 严格主实验：`src/strict_eval.py`、`slurm/strict_aoe_eval.sbatch`，Slurm job `466`
  在 NVIDIA A100-SXM4-80GB 完成；结果证据为 `results/strict/summary.json`、
  `results/strict/results.csv` 和 `results/strict/slurm-466.out`。

## 可核查实验编号

| 编号 | 设置 | 证据 |
|---|---|---|
| E1 | cosine 基线，CLS，seed=42 | `results/baseline_cosine_cls/` |
| E2 | AoE，CLS，cosine+angle，seed=42 | `results/aoe_cls/` |
| E3 | 仅 angle 目标，CLS，seed=42 | `results/ablation_angle_only_cls/` |
| E4 | AoE，mean pooling，seed=42 | `results/aoe_mean_pooling/` |
| E5 | AoE，CLS，seed=7 | `results/aoe_cls_seed7/` |

本人应能解释：复数除法如何得到 `re/im` 角度表示、为何要对高低标签做排序
损失、为何当前轻量设置没有复现原文的全部增益，以及冻结编码器和数据范围
对结论的影响。
