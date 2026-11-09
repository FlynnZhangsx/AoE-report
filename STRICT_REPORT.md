# AoE 严格复现记录（期中提交版）

## 结论先行

本次严格实验不再使用 BERT-Tiny 作为主结果，而是使用论文官方发布的
`SeanLee97/angle-bert-base-uncased-nli-en-v1` BERT-base checkpoint。该模型是
官方 AoE 实现训练的 SNLI+MultiNLI 模型；评测脚本按官方 `cls_avg` 池化和论文
STS 评价协议计算 Spearman 相关系数。GPU 作业已提交到 Slurm 并完成：job `466`
在 NVIDIA A100-SXM4-80GB 上运行成功，不是本地 CPU 的伪 GPU 结果。

## 严格对应关系

| 论文要求 | 本目录证据 |
|---|---|
| AoE 官方模型 | `models/angle-bert-base-uncased-nli-en-v1/` |
| 官方 pooling | `src/strict_eval.py` 中 `(CLS + attention-mask mean) / 2` |
| 标准 STS12–16、STS-B、SICK-R | `data/mteb/*-sts.jsonl.gz`、`data/stsbenchmark/test.parquet` |
| 长文本 GIS | `data/gis/test-00000-of-00001.parquet` |
| GPU 真实运行 | `slurm/strict_aoe_eval.sbatch`、`results/strict/slurm-466.out/.err` |
| 可核查数值 | `results/strict/results.csv`、`summary.json`、`*_predictions.npz` |

## GPU 严格评测结果

| 任务 | 样本数 | Spearman |
|---|---:|---:|
| STS12 | 3,108 | 0.7503 |
| STS13 | 1,500 | 0.8562 |
| STS14 | 3,750 | 0.8035 |
| STS15 | 3,000 | 0.8636 |
| STS16 | 1,186 | 0.8243 |
| SICK-R | 9,927 | 0.8151 |
| STS-B | 1,379 | 0.8546 |
| 标准 STS 七任务平均 | — | **0.8239** |
| GIS（官方 checkpoint 零样本） | 1,548 | 0.1650 |

结果与官方模型卡公布的 BERT AoE 结果接近：STS12–16、STS-B、SICK-R 分别为
75.09、85.56、80.66、86.44、82.47、85.16、81.23。GIS 的 0.1650 不是作业失败：
论文 Table 2 的 GIS 是在 GIS 训练集上进行的 in-domain 微调结果，本次 job 评测的
是官方 NLI checkpoint 的 zero-shot GIS，实验条件不同，不能混写。

GIS 测试集为 1,548 对（807 个 duplicate、741 个 non-duplicate），标签方向沿用
官方数据集：1 为 duplicate/similar，0 为 non-duplicate。论文 Table 2 对 GIS 与
STS-B 都报告 Spearman；脚本因此不把 GIS 改成准确率。

## 复现命令

```bash
cd /home/zsx/nlp
conda activate nlp
sbatch slurm/strict_aoe_eval.sbatch
# 监控
squeue -j 466
# 作业已结束，查看结果
cat results/strict/slurm-466.out
cat results/strict/summary.json
```

Slurm 作业参数是 1 张 GPU、16 CPU、64 GB 内存、2 小时；模型最大输入长度 512，
batch size 64。脚本会记录 Python、PyTorch、Transformers、GPU 型号、模型权重
SHA256、每个任务的样本数、耗时、Spearman，以及每一对样本的预测值。

## 与论文数字的边界

官方 checkpoint 评测是严格的“发布模型复现”；它不是重新从随机初始化训练一遍
BERT-base。若要求重新训练，必须额外等待 GPU 并按论文 SNLI+MultiNLI、seed=42、
学习率 5e-5 等配置运行；本次已用官方 checkpoint 完成全套标准 STS 评测。旧的
BERT-Tiny 投影实验仍保留在 `results/`，但明确标为受限/历史实验，不计入严格主结果。

