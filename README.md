# AoE: Angle-optimized Embeddings for Semantic Textual Similarity

本目录包含期中复现材料。严格主实验使用官方发布的 BERT-base AoE checkpoint，
并在 Slurm GPU 上评测 STS12–16、STS-B、SICK-R 和 GIS；入口是
`src/strict_eval.py` 和 `slurm/strict_aoe_eval.sbatch`。旧的 BERT-Tiny 投影
实验仅作为受限/历史对照，详见 `STRICT_REPORT.md`，不冒充论文严格结果。

## 环境

```bash
conda activate nlp
/home/zsx/.conda/envs/nlp/bin/python -c "import torch,transformers; print(torch.__version__, transformers.__version__)"
```

环境关键版本记录在 `ENVIRONMENT.md`。模型和数据已放在本目录；`data/` 总量约
648 MB，低于用户后来放宽的 1000 MB 上限。

关键 Python 依赖版本也列在 `requirements.txt`。

## 严格 GPU 重跑

```bash
conda activate nlp
sbatch slurm/strict_aoe_eval.sbatch
```

作业结束后，严格结果位于 `results/strict/summary.json` 和
`results/strict/results.csv`。按评分点撰写的完整报告见 `REPORT.md`，严格边界说明
见 `STRICT_REPORT.md`。

## 历史轻量实验

```bash
conda activate nlp
PYTHONUNBUFFERED=1 python src/aoe_repro.py \
  --poolings cls mean --epochs 12 --batch-size 256 --feature-batch-size 128 \
  2>&1 | tee results/console.log
MPLBACKEND=Agg python src/analyze_results.py
```

脚本会自动构造 `data/stsbenchmark/features/` 特征缓存；每个实验目录保存
`training.jsonl`、`best.pt` 和 `result.json`，汇总为 `results/results.csv`。

## 文件索引

- `REPORT.md`：按 `rule.xlsx` 评分点撰写的期中实验报告。
- `src/aoe_repro.py`：特征提取、AoE 角度损失、训练和评测。
- `src/analyze_results.py`：子集分析和验证曲线生成。
- `data/stsbenchmark/`：带标签 STSBenchmark parquet。
- `models/bert-tiny/`：本地 BERT-Tiny 权重。
- `results/`：真实控制台输出、逐 epoch 日志、模型检查点、结果表和图。
