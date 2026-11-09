# AoE: Angle-optimized Embeddings for Semantic Textual Similarity期中实验报告

本次期中作业复现 ACL 2024 论文 AoE 的核心思想：普通余弦相似度在饱和区梯度变小，模型
难以区分表面词汇高度重叠、但语义关系不同的句对；AoE 将句向量划分为实部和
虚部，在复数空间通过除法获得归一化角度差，再用角度排序目标与余弦排序对比目标
联合优化。

使用官方发布的 `SeanLee97/angle-bert-base-uncased-nli-en-v1` BERT-base AoE checkpoint，
在 NVIDIA A100-SXM4-80GB 上运行，按官方 `cls_avg` 池化方式评测 STS12–16、
STS-B、SICK-R 和 GIS。STS 七任务平均 Spearman 为 **0.8239**；逐任务为：STS12 0.7503、
STS13 0.8562、STS14 0.8035、STS15 0.8636、STS16 0.8243、SICK-R 0.8151、
STS-B 0.8546。官方模型卡对应数字的平均值为 0.8237，差值约 0.0002。

---

## 1. 论文与复现范围

### 1.1 论文要解决的问题

论文指出，许多句向量模型把 cosine 作为主要训练目标。当两个向量已经接近同向或
反向时，cosine 位于饱和区，继续调整模型参数对 cosine 的影响很小，梯度也会变弱。
这在 SNLI、MultiNLI 等 NLI 数据中尤其重要：中性句对可能与蕴含句对共享大量表面
词汇，模型却容易只利用词汇重叠而忽略隐含语义差异。

论文的核心假设是：若将句向量放入复数空间，复数除法产生的相位或角度差可以提供不同于普通 cosine的细粒度信号，因此在 cosine 饱和区仍然能够区分相似度排序。

### 1.2 原论文实验范围

| 类别 | 论文内容 | 指标/用途 |
|---|---|---|
| 训练数据 | SNLI + MultiNLI | 监督句对训练 |
| 标准短文本 STS | STS12、STS13、STS14、STS15、STS16、STS-B、SICK-R | Spearman |
| 长文本 STS | GitHub Issue Similarity Dataset（GIS） | Spearman |
| 下游分类 | MR、CR、SUBJ、MPQA、SST-2、TREC、MRPC | SentEval accuracy |
| MTEB | classification、clustering、pair classification、reranking、retrieval、STS、summarization | 官方 MTEB 指标 |
| 分析 | cosine 饱和区、实部/虚部距离、预测分布 | Figure 3–5 等 |

论文材料与本目录证据对应如下：

| 论文材料 | 论文结论 | 本次对应证据 |
|---|---|---|
| Figure 1 | cosine 饱和区中的表面相似/语义不同案例 | 第 6 节机制解释 |
| Figure 2 | Encoder→复数句向量→complex division→angle ranking | `src/aoe_repro.py` 的 `Projection`、`angle_scores`、`angle_loss` |
| Table 1 | 七个标准 STS 数据集上的模型比较 | 官方 checkpoint Slurm 结果与模型卡数值对照 |
| Table 2 | STS-B/GIS 的 in-domain 训练结果 | 论文数值对照，并区分 zero-shot 与 in-domain |
| Table 3 | 七项下游分类任务 | 官方 SentEval 结果与论文数值对照 |
| Table 4 | objective、pooling、random seed 消融 | E1–E5 同条件消融日志 |
| Figure 3–5 | 饱和区中实部/虚部差异和预测分布 | `subset_metrics.csv`、验证曲线和文字分析 |

### 1.3 本次期中锁定的复现范围

| 范围 | 状态 | 说明 |
|---|---|---|
| 官方 AoE BERT-base checkpoint 推理 | **已完成** | A100、真实日志和逐样本预测 |
| STS12–16、STS-B、SICK-R | **已完成** | 7 个标准 STS 任务，Spearman |
| GIS test zero-shot | **已完成** | 1,548 对，Spearman；不是 Table 2 in-domain |
| AoE 核心公式与排序损失 | **已完成** | 复数除法、角度目标、cosine 辅助目标 |
| 同条件 cosine/AoE/angle/pooling/seed 消融 | **已完成** | E1–E5，逐 epoch 日志 |
| STS-B/GIS in-domain 微调 | **已完成** | 官方 Format-A、`cls_avg`、完整 train/valid/test；结果见 `results/in_domain_strict/` |
| 7 个下游任务 | **已完成** | 官方 SentEval、默认分类器；结果见 `results/downstream/official_senteval.json` |

---

## 2. 方法原理与实现

### 2.1 复数句向量

给定句子 `x`，Transformer 编码器产生维度为 `2d` 的句向量：

```text
X = Encoder(x) ∈ R^(2d)
X = [X^re ; X^im]
```

前 `d` 维作为实部，后 `d` 维作为虚部。句向量写成：

```text
z = a + b i,   a = X^re, b = X^im
w = c + d i,   c = Y^re, d = Y^im
```

论文中 BERT 使用 CLS 句级向量；官方实现还提供 `cls_avg` 等池化方式。本次推理使用官方模型卡指定的 `cls_avg`：

```text
avg = Σ(token_embedding * attention_mask) / Σ(attention_mask)
embedding = (CLS + avg) / 2
```

### 2.2 Complex division 与角度差

对两个复数向量做除法：

```text
(a + b i) / (c + d i)
= ((a c + b d) + (b c - a d)i) / (c² + d²)
```

实现为：

```python
denom = (c.square() + d.square()).sum(dim=1, keepdim=True)
quotient_re = (a * c + b * d) / denom
quotient_im = (b * c - a * d) / denom
```

之后用两侧向量幅度归一化：

```text
|z| = sqrt(sum(a² + b²))
|w| = sqrt(sum(c² + d²))
re ← re / (|z| / |w|)
im ← im / (|z| / |w|)
```

官方 `sum` pooling 的顺序是先对归一化后的实部和虚部求和，再取绝对值：

```text
Δθ(z,w) = | sum(re_normalized + im_normalized) |
```

### 2.3 CoSENT 风格排序损失

一个 batch 内有句对 `(x_i, x_i')` 和人工相似度 `s_i`。如果 `s_i < s_j`，模型
输出应满足 `q_i < q_j`：

```text
L_rank(q,s) = log(1 + Σ_{s_i < s_j} exp(τ(q_i - q_j)))
```

代码用 `logsumexp` 实现，并用 mask 排除不满足排序关系的项。这避免把 STS 的连续
分数粗暴离散化，保留人工分数的排序信息。

### 2.4 联合目标与消融

轻量方法实验使用：

```text
L_cosine = ranking_loss(cosine(x_i, x_i'), s_i)
L_angle  = ranking_loss(angle(x_i, x_i'), s_i)
L_AoE    = L_cosine + 0.02 * L_angle
```

对应三个 objective：cosine baseline 只使用 `L_cosine`；AoE 使用联合目标；
angle-only 只使用 `0.02 L_angle`。官方仓库训练示例也使用 `angle_w=0.02`、
`cosine_tau=20`、`angle_tau=20`；我们的实验中固定这些参数，保证消融可比。

---

## 3. 数据、预处理与环境

### 3.1 主实验数据

| 数据 | 文件 | 样本数 | 用途 |
|---|---|---:|---|
| STS12 | `data/mteb/sts12-sts.jsonl.gz` | 3,108 | 标准 STS |
| STS13 | `data/mteb/sts13-sts.jsonl.gz` | 1,500 | 标准 STS |
| STS14 | `data/mteb/sts14-sts.jsonl.gz` | 3,750 | 标准 STS |
| STS15 | `data/mteb/sts15-sts.jsonl.gz` | 3,000 | 标准 STS |
| STS16 | `data/mteb/sts16-sts.jsonl.gz` | 1,186 | 标准 STS |
| SICK-R | `data/mteb/sickr-sts.jsonl.gz` | 9,927 | 标准 STS |
| STS-B | `data/stsbenchmark/test.parquet` | 1,379 | 标准 STS |
| GIS | `data/gis/test-00000-of-00001.parquet` | 1,548 | 长文本 zero-shot |

GIS train/valid 文件保存在 `data/gis/`，用于本实验的 in-domain 训练。GIS 官方划分
为 train 18,565、valid 1,547、test 1,548；test 中 duplicate 为 807、
non-duplicate 为 741，标签 1 表示 duplicate/similar，0 表示 non-duplicate。

### 3.2 受限同条件实验数据

STS-B train/validation/test 分别为 5,749、1,500、1,379 对。文本只在特征提取时
tokenizer 截断，最大长度 64。

下游数据按 SentEval 原始目录组织在 `data/senteval/downstream/`：MR、CR、SUBJ、MPQA
使用正负文本文件，SST-2 使用 train/dev/test，TREC 使用 train_5500/TREC_10，MRPC
使用 3,668 条 train 和 1,725 条 test，并由 `src/prepare_mrpc.py` 转为 SentEval TSV。

### 3.3 模型与环境

模型目录：`models/angle-bert-base-uncased-nli-en-v1/`，hidden size 768、
12 层、最大输入 512、pooling 为 `cls_avg`。环境如下：

| 组件 | 版本 |
|---|---:|
| Python | 3.10.22 |
| PyTorch | 2.1.0+cu121 |
| Transformers | 4.41.2 |
| Datasets | 2.19.1 |
| NumPy | 1.25.1 |
| Pandas | 2.3.3 |
| SciPy | 1.15.3 |
| PyArrow | 25.0.1 |
| GPU | NVIDIA A100-SXM4-80GB |

---

## 4. 实验协议与公平对照

### 4.1 checkpoint 评测

固定：

```text
device       cuda
GPU          1 × NVIDIA A100-SXM4-80GB
batch_size   64
max_length   512
pooling      cls_avg
metric       Spearman correlation
```

每个数据集分别编码两侧文本，以向量 cosine 作为预测相似度，再与人工分数计算
Spearman。脚本保存 labels 和 predictions，避免只留最终数字。

### 4.2 同条件受限实验协议

| 编号 | 目的 | Encoder | 可训练部分 | objective | pooling | seed |
|---|---|---|---|---|---|---:|
| E1 | cosine baseline | frozen BERT-Tiny | 128→128 投影 | cosine | CLS | 42 |
| E2 | AoE 主实验 | frozen BERT-Tiny | 128→256 复数投影 | cosine+angle | CLS | 42 |
| E3 | angle 消融 | frozen BERT-Tiny | 128→256 复数投影 | angle-only | CLS | 42 |
| E4 | pooling 消融 | frozen BERT-Tiny | 128→256 复数投影 | cosine+angle | mean | 42 |
| E5 | seed 消融 | frozen BERT-Tiny | 128→256 复数投影 | cosine+angle | CLS | 7 |

共同配置：12 epochs、batch 256、AdamW、lr 1e-2、weight decay 1e-4、
angle weight 0.02、angle tau 20、max length 64。只用 validation Spearman
选择最优 epoch，test 只在选模后评估一次。

### 4.3 论文 Table 2/3 实验配置

Table 2 使用 STS-B 和 GIS 各自的完整 train/validation/test 划分，seed=42，
5 epochs，初始学习率 5e-5，AdamW，weight decay 0.01，官方 Format-A 的
`in_batch_negative + 0.02 * angle` 目标；STS-B 使用 batch 32、最大长度 128，
GIS 使用 batch 8、最大长度 512，均用 `cls_avg`，按 validation Spearman 选最佳 epoch。

Table 3 使用官方 SentEval engine，七个任务均采用默认 PyTorch 线性分类器、10-fold
正则搜索、seed=42、`cls_avg` embedding 和最大长度 512，记录真实 accuracy。

---

## 5. 实验结果与对照

### 5.1 严格 GPU 主结果

Slurm stdout 记录：

```text
HOST=hjadmin-4U-GPU-Server
GPU 0: NVIDIA A100-SXM4-80GB
device=cuda
torch=2.1.0+cu121
transformers=4.41.2
```

| 任务 | n | 论文 Table 1 AoE-BERTbase | 本实验 Spearman | 差值（本实验−论文） |
|---|---:|---:|---:|---:|
| STS12 | 3,108 | 0.7526 | **0.7503** | -0.0023 |
| STS13 | 1,500 | 0.8561 | **0.8562** | +0.0001 |
| STS14 | 3,750 | 0.8064 | **0.8035** | -0.0029 |
| STS15 | 3,000 | 0.8636 | **0.8636** | 0.0000 |
| STS16 | 1,186 | 0.8251 | **0.8243** | -0.0008 |
| SICK-R | 9,927 | 0.8099 | **0.8151** | +0.0052 |
| STS-B | 1,379 | 0.8564 | **0.8546** | -0.0018 |
| **七任务平均** | — | **0.8243** | **0.8239** | **-0.0004** |
| GIS zero-shot | 1,548 | — | **0.1650** | — |

论文 Table 1 的 AoE-BERTbase 平均为 82.43，本实验为 82.39，差距为 -0.04 个百分点。
官方模型卡的 BERT AoE 数字为 STS12 75.09、STS13 85.56、STS14 80.66、STS15
86.44、STS16 82.47、STS-B 85.16、SICK-R 81.23，平均 82.37；本实验平均 82.39，
与模型卡也仅差 0.02 个百分点。两套参考数字均与本实验接近，说明 tokenizer、池化、
cosine 和 Spearman 实现没有出现大尺度偏差；逐任务的小差异主要来自论文多次运行均值、
发布 checkpoint 与论文训练权重的差别，以及评测实现的版本差异。

### 5.2 论文 Table 2、Table 3、Table 4 对照

论文 Table 2 的 in-domain 对照为：

| 模型 | STS-B | GIS | 平均 |
|---|---:|---:|---:|
| SimCSE | 76.27±0.23 | 60.38±0.18 | 68.33 |
| SBERT | 84.67±0.35 | 69.50±0.47 | 77.09 |
| AoE | **86.28±0.19** | **70.59±0.35** | **78.44** |

论文 Table 2 的数字来自 STS-B/GIS 训练集上的 in-domain 训练，不能与官方 NLI
checkpoint 的 GIS zero-shot 0.1650 直接比较。本实验的 in-domain 结果如下：

| 初始化/设置（Spearman × 100） | STS-B | GIS | 平均 |
|---|---:|---:|---:|
| 论文 AoE-BERTbase | 86.28±0.19 | 70.59±0.35 | 78.44 |
| 本实验 released AoE checkpoint、seed=42 | **82.26** | **63.54** | **72.90** |
| 差值（本实验−论文） | **-4.02** | **-7.05** | **-5.54** |

本实验严格使用论文数据划分、AoE Format-A 目标和 validation 选模，
结果低于论文，主要原因是初始化使用已发布 NLI checkpoint 而非原始
`bert-base-uncased`，并且本实验使用单一 seed，而论文报告多次运行的均值和标准差。
对应结果文件中的原始 Spearman 为 STS-B=0.822575、GIS=0.635394。

论文 Table 3 的 AoE-BERTbase 七项下游平均 accuracy 为 86.96；本实验使用官方
SentEval 得到：

| 任务 | 论文 AoE-BERTbase | 本实验 accuracy | 差值（本实验−论文） |
|---|---:|---:|---:|
| MR | 83.00 | 82.55 | -0.45 |
| CR | 89.38 | 88.98 | -0.40 |
| SUBJ | 94.72 | 94.89 | +0.17 |
| MPQA | 89.87 | 89.88 | +0.01 |
| SST-2 | 87.20 | 87.59 | +0.39 |
| TREC | 89.00 | 90.80 | +1.80 |
| MRPC | 75.54 | 77.51 | +1.97 |
| **七任务平均** | **86.96** | **87.46** | **+0.50** |

MRPC 另记录 F1=84.41。七项平均比论文高 0.50 个百分点，说明在当前
`cls_avg`、seed=42 和 SentEval 默认分类器设置下，下游迁移总体接近论文；MR、CR
略低，主要与单 seed 和 checkpoint 初始化差异有关，TREC、MRPC 的提升则来自任务
划分和分类器正则选择的随机性。

论文 Table 4 的标准 STS 消融为：AoE 82.43±0.08、only angle 82.36±0.14、
only contrastive 81.53±0.19；pooling=cls 82.43±0.11、avg 81.69±0.18、
max 77.96±0.21。

### 5.3 同条件 E1–E5 结果

| 实验 | objective | pooling | seed | best epoch | validation ρ | test ρ |
|---|---|---|---:|---:|---:|---:|
| E1 cosine baseline | cosine | CLS | 42 | 4 | 0.7444 | **0.6509** |
| E2 AoE | cosine + angle | CLS | 42 | 7 | 0.7395 | **0.6494** |
| E3 angle-only | angle | CLS | 42 | 12 | 0.7388 | **0.6465** |
| E4 pooling ablation | cosine + angle | mean | 42 | 8 | 0.7530 | **0.6608** |
| E5 seed ablation | cosine + angle | CLS | 7 | 5 | 0.7387 | **0.6440** |

论文 Table 4 的 AoE 平均为 82.43（百分制），本实验 E2 为 64.94，差距为
-17.49 个百分点；E4 mean pooling 为 66.08，仍低于论文 avg pooling 的 81.69。
差距来自 BERT-Tiny 冻结 encoder、只训练小投影头、只使用 STS-B 数据，以及论文的
SNLI+MultiNLI 训练和多 seed 统计设置。

### 5.4 饱和区和子集核查

STS-B test 后验子集分析：

| 子集 | n | E1 cosine | E2 AoE | E2−E1 |
|---|---:|---:|---:|---:|
| 全部 | 1,379 | 0.6509 | 0.6494 | -0.0016 |
| 高相似度 `y>=4` | 338 | 0.3796 | 0.3793 | -0.0002 |
| 低相似度 `y<=2` | 534 | 0.4983 | 0.4892 | -0.0092 |

本实验轻量设置没有显示 AoE 优势，反而说明只训练小投影头、只用 STS-B 连续标签，
不能自动获得论文 NLI 训练设置中的增益。

可视化和原始统计文件：[`validation_curves.png`](/home/zsx/nlp/results/validation_curves.png)、
[`subset_metrics.csv`](/home/zsx/nlp/results/subset_metrics.csv)。

---

## 6. 结果分析、差异解释与局限

### 6.1 严格主结果的有效性

严格结果满足四个条件：模型一致、池化一致、任务一致、证据完整。具体而言，使用
官方 BERT AoE 权重、官方 `cls_avg`、七个标准 STS 和 GIS test，stdout 记录 A100、
CUDA、版本、样本数和 Spearman，结果文件保存每一对 prediction。本地平均 0.8239
与官方模型卡平均 0.8237 的差异约 0.02 个百分点。

### 6.2 GIS zero-shot 为什么低

GIS 包含 issue 描述、代码、HTML 标记和长文本。官方 NLI checkpoint 主要在短句 NLI
上训练；zero-shot 设定 max length 512 时 Spearman 只有 0.1650，反映的是长文本迁移，
不是论文 Table 2 的 in-domain AoE。本实验使用完整 GIS train/valid/test 和 AoE 目标
微调后，测试 Spearman 提升到 0.6354，说明训练数据确实解决了部分域差异；但仍低于
论文 0.7059，主要受初始化不同和单 seed 限制。

### 6.3 轻量 AoE 为什么没有超过 baseline

E2 比 E1 低 0.0016，E3 比 E1 低 0.0045，主要差异是：论文使用 BERT-base、
LLaMA-7B/13B，而本实验是 2 层、128 维 BERT-Tiny；论文使用 SNLI+MultiNLI，而本实验
只使用 STS-B 连续分数；论文训练 backbone，而本实验冻结 encoder；论文 grid search
并报告多次运行均值，而本实验固定单一训练预算；E5 还显示 seed 影响（0.6440 vs
0.6494）。

### 6.4 pooling、seed 和资源

E4 mean pooling 得到 0.6608，超过 CLS AoE 的 0.6494，说明 pooling 结论依赖于
backbone、数据和训练目标。E5 的 seed=7 低于 seed=42，不能只跑一个 seed 就声称
鲁棒。严格 job 总运行约 1 分钟：短文本任务约 1–3 秒，SICK-R 约 6.6 秒，GIS
约 29.6 秒；GPU 只用于官方 checkpoint 推理，没有把 CPU 轻量训练伪装成 GPU 结果。

---

## 7. 代码、数据、环境与完整复现步骤

### 7.1 目录结构

```text
/home/zsx/nlp/
├── rule.xlsx
├── REPORT.md / STRICT_REPORT.md / README.md / ENVIRONMENT.md / CONTRIBUTION.md
├── src/aoe_repro.py                  # 轻量 AoE 与消融
├── src/analyze_results.py            # 子集和曲线分析
├── src/strict_eval.py                # 官方 checkpoint 严格评测
├── src/in_domain_train.py            # STS-B/GIS in-domain 实验
├── src/senteval_downstream.py        # 七项官方 SentEval 下游任务
├── src/prepare_mrpc.py               # MRPC 数据格式转换
├── slurm/strict_aoe_eval.sbatch      # A100 作业脚本
├── slurm/in_domain_aoe.sbatch        # STS-B/GIS 实验
├── slurm/downstream_senteval.sbatch  # SentEval 实验
├── models/                           # BERT-Tiny 与官方 AoE BERT
├── data/                             # STS、GIS、NLI、SentEval 数据
└── results/                          # 严格结果、日志、模型和图
```

### 7.2 GPU 实验运行

```bash
cd /home/zsx/nlp
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate nlp
sbatch slurm/strict_aoe_eval.sbatch
```

脚本申请 1 张 GPU、16 CPU、64 GB 内存；执行 `nvidia-smi -L`；调用
`strict_eval.py --device cuda --batch-size 64 --max-length 512`；最后将 stdout/stderr
写入 `results/strict/slurm-%j.out/.err`。

### 7.3 结果核查

```bash
cat results/strict/slurm-466.out
cat results/strict/slurm-466.err
cat results/strict/summary.json
cat results/strict/results.csv
```

### 7.4 轻量实验运行

```bash
cd /home/zsx/nlp
conda activate nlp
PYTHONUNBUFFERED=1 python src/aoe_repro.py \
  --poolings cls mean --epochs 12 --batch-size 256 \
  --feature-batch-size 128 2>&1 | tee results/console.log
MPLBACKEND=Agg python src/analyze_results.py
```

### 7.5 实验运行命令

```bash
cd /home/zsx/nlp
conda activate nlp
jid=$(sbatch --parsable slurm/in_domain_aoe.sbatch)
python src/prepare_mrpc.py
sbatch --dependency=afterok:${jid} slurm/downstream_senteval.sbatch
```

最小检查：

```bash
python -m py_compile src/aoe_repro.py src/strict_eval.py src/in_domain_train.py \
  src/senteval_downstream.py src/prepare_mrpc.py
python - <<'PY'
import pandas as pd
print(pd.read_parquet('data/stsbenchmark/test.parquet').shape)
print(pd.read_parquet('data/gis/test-00000-of-00001.parquet').shape)
PY
```

### 7.6 关键 SHA256

```text
models/angle-bert-base-uncased-nli-en-v1/pytorch_model.bin
6d2e2809bd060555dbe98df9a69d4832c7c4c54892a6b33580b9199f6579cd02

data/stsbenchmark/test.parquet
4f7599f3d9de73bdfd672833acadb8fef42fa748c86b409b07873f780a405fe4

data/gis/test-00000-of-00001.parquet
d2866f4daf59dbcdf0ac7098adc886ccf711e4c88bbc98c6ea7de66d2fdafb4d

src/strict_eval.py
9c4d2989af0fce53ec2eaaede9403876e7fe8936367b58afa82b91c483242818
```

---

## 9. 个人实际工作与理解

### 9.1 实际产出

- 阅读 ACL 原文，整理问题、公式、数据、表格和评价指标；
- 实现特征缓存、复数投影、complex division、angle ranking loss、训练/验证/测试；
- 实现高低相似度子集统计和 validation 曲线；
- 下载整理 STS-B、STS12–16、SICK-R、GIS、SNLI/MultiNLI 和模型；
- 创建 `nlp` conda 环境，固定 Python/PyTorch/Transformers 版本；
- 编写严格评测脚本和 Slurm 脚本，在 A100 上完成官方 checkpoint 评测；
- 按官方 Format-A 的 zigzag 句对排列完成 STS-B/GIS in-domain 训练；
- 整理 SentEval 的 MR、CR、SUBJ、MPQA、SST-2、TREC、MRPC 数据，完成七项下游评测；
- 保存逐任务分数、逐样本预测、运行时间、GPU 信息和模型 SHA256；
- 将 BERT-Tiny 受限结果与官方 checkpoint 结果分层。

### 9.2 关键理解

**为什么需要虚部？** 实部延续普通句向量的表面语义表示，虚部提供额外自由度，
用于表示 cosine 饱和区中难以表达的细粒度差异。重点不是简单扩大维度，而是通过
复数除法得到相对角度信息。

**为什么使用排序损失而不是 MSE？** STS 评价关注排序相关性。CoSENT 风格损失直接
惩罚“低标签句对分数高于高标签句对”，利用连续标签的排序关系，与论文 objective
一致。

**为什么 GIS zero-shot 低不等于实现错误？** 官方 NLI checkpoint 主要在短句 NLI
上训练，GIS zero-shot 与 GIS train 上的 in-domain 微调不是同一设置；本实验分别报告
0.1650 和 0.6354，不能把两者或论文 Table 2 的 0.7059 混成一个结果。

**BERT-Tiny 与论文设置的差异是什么？** 本实验使用 2 层、128 维并冻结 encoder，
只训练一个小投影头；论文训练 BERT-base/LLM，并以 SNLI+MultiNLI 为主要训练数据，
因此 E1–E5 只用于验证方法机制，不能作为论文绝对指标的替代。

---

## 参考文献与数据链接

1. Li, Xianming and Li, Jing. 2024. *AoE: Angle-optimized Embeddings for Semantic Textual Similarity*. ACL 2024.  
   [ACL Anthology](https://aclanthology.org/2024.acl-long.101/) · [PDF](https://aclanthology.org/2024.acl-long.101.pdf)
2. SeanLee97. [AnglE 官方仓库](https://github.com/SeanLee97/AnglE)。
3. [官方 BERT AoE 模型卡](https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1)。
4. [STS12](https://huggingface.co/datasets/mteb/sts12-sts)、[STS13](https://huggingface.co/datasets/mteb/sts13-sts)、
   [STS14](https://huggingface.co/datasets/mteb/sts14-sts)、[STS15](https://huggingface.co/datasets/mteb/sts15-sts)、
   [STS16](https://huggingface.co/datasets/mteb/sts16-sts)、[SICK-R](https://huggingface.co/datasets/mteb/sickr-sts)。
5. [GIS: WhereIsAI/github-issue-similarity](https://huggingface.co/datasets/WhereIsAI/github-issue-similarity)。
