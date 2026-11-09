# 可复现环境

环境名称：`nlp`  
解释器：`/home/zsx/.conda/envs/nlp/bin/python`  
Python：3.10.22  
登录节点没有 GPU 驱动；Slurm GPU 节点使用同一环境运行严格实验。作业脚本会在
分配到 A100 后检查 `torch.cuda.is_available()` 和 `nvidia-smi -L`，若 CUDA 不可见
则直接失败，不会把 CPU 结果伪装成 GPU 结果。

关键依赖：

| 包 | 版本 |
|---|---:|
| torch | 2.1.0+cu121 |
| transformers | 4.41.2 |
| datasets | 2.19.1 |
| huggingface-hub | 0.23.4 |
| sentencepiece | 0.2.0 |
| numpy | 1.25.1 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.5.0 |
| matplotlib | 3.9.0 |
| pyarrow | 25.0.1 |

环境创建过程：先将本机已有的 Python 3.10 conda 环境离线克隆为 `nlp`，
再在 `nlp` 中安装 Transformers/Datasets/SentencePiece/Hugging Face 依赖：

```bash
conda create -n nlp --clone /home/zsx/.conda/envs/fastavatar -y
/home/zsx/.conda/envs/nlp/bin/pip install \
  transformers==4.41.2 datasets==2.19.1 sentencepiece==0.2.0 huggingface_hub==0.23.4
```

关键文件 SHA256：

```text
config.json       cf3179684f7c360dd6185962d480e7d426ceb8fe49f5db7b64bdc15dba0b8798
pytorch_model.bin dab2c2bddcfb48ea430ef63fd76d46d67d704487844d967256a50dd7d7fd0a66
vocab.txt          07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
train.parquet      539e96b55b1ff809588d3be46cdcea07c0940636b9579d79ebf447a7550cfe85
validation.parquet daa201c427073036d99cbfadf06cf64a52cfd648d9383260408001c67ace8d9f
test.parquet       4f7599f3d9de73bdfd672833acadb8fef42fa748c86b409b07873f780a405fe4
```

严格主模型的权重为：

```text
models/angle-bert-base-uncased-nli-en-v1/pytorch_model.bin
sha256: 6d2e2809bd060555dbe98df9a69d4832c7c4c54892a6b33580b9199f6579cd02
```
