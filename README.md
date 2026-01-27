# 实验五：多模态情感分类

## 项目简介


## 环境配置
### 1. 创建虚拟环境
```bash
conda create -n multimodal python=3.11
conda activate multimodal
```

### 2. 下载依赖
本实现基于 Python3。运行代码需要安装依赖（见 `requirements.txt`），你可以直接执行：

```bash
pip install -r requirements.txt
```

## 数据集说明

将课程提供的数据集放置在 `project5/` 下：

- `project5/train.txt`（CSV：`guid,tag`）
- `project5/test_without_label.txt`（CSV：`guid,tag`，其中 tag 为 `null`）
- `project5/data/{guid}.txt`
- `project5/data/{guid}.jpg`

请勿将原始 `project5/data/` 数据文件提交到仓库。

## Baseline

基线模型：

- 文本编码器：`google-bert/bert-base-uncased`
- 图像编码器：`resnet50`
- 融合方式：特征拼接 + MLP 分类器

训练/验证输出指标（验证集）：

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 (macro)

同时记录模型参数量（total/trainable），并保存 loss 曲线与混淆矩阵。

## Repository structure

我们列出仓库中主要文件的结构说明：

```
|-- project5/
|   |-- train.txt
|   |-- test_without_label.txt
|   |-- data/
|       |-- {guid}.txt
|       |-- {guid}.jpg
|-- src/
|   |-- __init__.py
|   |-- datasets.py          # 数据读取与 Dataset
|   |-- train.py             # 训练入口（划分验证集、保存 best ckpt、输出曲线/矩阵）
|   |-- evaluate.py          # 可选：对保存的 val split 做独立评估
|   |-- predict.py           # 预测测试集并生成结果文件（替换 null）
|   |-- utils.py             # seed、参数量统计、json 保存等
|   |-- models/
|       |-- __init__.py
|       |-- baseline.py      # twitter-roberta-base + resnet50
|-- outputs/                 # 训练输出目录（运行后生成）
|-- requirements.txt
|-- README.md
```

## Run pipeline

### 1) 训练（自动划分验证集）

在仓库根目录运行：

```bash
python -m src.train --data_root project5 --output_root outputs --epochs 6 --batch_size 16 --max_length 128
```

训练输出会保存在 `outputs/run_*/` 下：

- `checkpoints/best.pt`
- `plots/loss_curve.png`
- `plots/confusion_matrix.png`
- `model_params.json`（模型参数量：total/trainable）
- `summary.json`（最佳 epoch、最佳指标、loss 历史）
- `logs/epoch_*.json`

### 2) 评估（可选）

对已保存的验证集划分与 best checkpoint 进行评估：

```bash
python -m src.evaluate --data_root project5 --ckpt outputs/run_YYYYMMDD_HHMMSS/checkpoints/best.pt --split_json outputs/run_YYYYMMDD_HHMMSS/split.json --output_dir outputs/run_YYYYMMDD_HHMMSS/eval
```

### 3) 测试集预测（生成提交文件）

对 `test_without_label.txt` 进行预测，并输出 `guid,tag`（将 `tag` 替换为模型预测结果）：

```bash
python -m src.predict --data_root project5 --ckpt outputs/run_YYYYMMDD_HHMMSS/checkpoints/best.pt --output_path outputs/run_YYYYMMDD_HHMMSS/pred_test.txt
```

## Attribution

本仓库使用/参考了以下开源库的实现思路：

- PyTorch
- torchvision
- HuggingFace Transformers
- scikit-learn