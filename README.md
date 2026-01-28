# 实验五：多模态情感分类

## 1. 项目简介

本项目实现了一个用于社交媒体场景的 **多模态情感分类系统**，输入为每条样本对应的文本（`guid.txt`）和图像（`guid.jpg`），输出三类情感标签之一：`negative`、`neutral`、`positive`。

核心特性包括：

- 文本分支：基于预训练 `google-bert/bert-base-uncased`，支持噪声清洗与轻量文本增强；
- 图像分支：支持 `ResNet50` 与 `DenseNet121` 两种骨干网络，并提供基础/轻量/强增强三档图像增强策略；
- 融合方式：在统一框架下实现 Concat、加权和、门控、注意力、双线性交互等多种多模态融合；
- 训练策略：分层冻结、分组学习率、余弦退火调度与早停，配合系统化的消融实验（Text-only / Image-only / Multi-Base / Multi-Best）。

基线配置为：**BERT 文本编码器 + ResNet50 图像编码器 + Concat 融合**，在此基础上展开后续所有对比实验。

---

## 2. 目录结构

仓库主要目录与文件如下：

```text
multimodal-sentiment-classification/
├── project5/
│   ├── train.txt                 # 训练集标注（guid,tag）
│   ├── test_without_label.txt    # 测试集（guid,tag=null）
│   └── data/
│       ├── {guid}.txt            # 文本文件
│       └── {guid}.jpg            # 图像文件
├── src/
│   ├── __init__.py
│   ├── datasets.py               # 数据集定义、CSV 读取、文本/图像加载
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py           # 多模态基线模型与融合策略
│   ├── train.py                  # 训练入口：划分验证集、训练、保存模型
│   ├── evaluate.py               # 独立评估脚本：加载模型并在验证集上评估
│   ├── predict.py                # 测试集推理：生成提交文件
│   ├── show_bad_cases.py         # 从验证集抽取 bad cases 进行误判分析
│   └── utils.py                  # 随机种子、参数统计、ModelScope 支持等
├── outputs/                      # 训练与评估输出目录（运行后自动生成）
├── requirements.txt              # Python 依赖列表
└── README.md
```

---

## 3. 环境配置

### 3.1 创建虚拟环境

推荐使用 Conda 创建独立环境（Python ≥ 3.10）：

```bash
conda create -n multimodal python=3.11
conda activate multimodal
```

### 3.2 安装依赖

直接在仓库根目录执行：

```bash
pip install -r requirements.txt
```

如果需要手动安装，主要依赖包括：
- torch, torchvision

- transformers

- scikit-learn, pandas, numpy

- matplotlib, Pillow

- tqdm

- modelscope（国内用户加速下载）

如果在国内网络环境下从 HuggingFace 下载模型较慢，项目会优先使用 ModelScope 进行下载缓存，可通过环境变量关闭：

```bash
set DISABLE_MODELSCOPE=1  
```

---

## 4. 命令行参数与运行方式

### 4.1 训练入口：`src.train`

在仓库根目录运行（下列命令对应多模态基线模型配置：BERT + ResNet50 + Concat，默认无额外清洗与图像增强）

```bash
python -m src.train \
  --data_root project5 \
  --output_root outputs \
  --epochs 10 \
  --batch_size 16 \
  --max_length 128 \
  --image_aug base \
  --image_backbone resnet50 \
  --fusion concat \
  --mode multimodal
```

常用参数说明：

- **数据与训练控制**
  - `--data_root`：数据根目录（默认 `project5`，你需要换成你自己的数据存放路径）。
  - `--output_root`：实验输出根目录（默认 `outputs`，下属 `run_*` 子目录自动创建）。
  - `--seed`：随机种子（默认 42）。
  - `--epochs`：最大训练轮数（默认 10）。
  - `--batch_size`：批大小（默认 16）。
  - `--max_length`：文本最大长度（默认 128）。
  - `--lr`：基础学习率（默认 `2e-5`）。
  - `--weight_decay`：权重衰减系数（默认 `0.01`）。
  - `--resnet_lr_scale`：图像分支学习率缩放系数（默认 `0.1`）。
  - `--early_stop_patience`：早停耐心轮数（默认 3，按验证集 Macro F1）。
  - `--val_size`：验证集比例（默认 0.2，分层划分）。

- **文本配置（`--text_config`）**：控制文本预处理/增强与编码器类型
  - 可选值：`base`、`clean`、`aug`、`twitter`、`all`
  - `base`(默认)：仅解码与截断，直接送入 BERT。
  - `clean`：启用噪声归一化（URL/@user/emoji/重复标点）。
  - `aug`：在 `clean` 基础上加入轻量文本增强（随机删除少量非关键 token）。
  - `twitter`：使用 `cardiffnlp/twitter-roberta-base-sentiment-latest` 作为编码器。
  - `all`：同时启用 `clean` 与 `aug`，并使用 Twitter-RoBERTa 编码器。

- **图像增强（`--image_aug`）**：控制图像预处理强度
  - 可选值：`base`、`light`、`strong`
  - `base`(默认)：仅 `Resize + ToTensor + Normalize`，无随机增强。
  - `light`：`RandomResizedCrop(0.8-1.0) + HorizontalFlip + 轻度 ColorJitter`。
  - `strong`：更大范围的 `RandomResizedCrop(0.6-1.0)` 与更强 ColorJitter。

- **图像骨干网络（`--image_backbone`）**
  - 可选值：`resnet50`(默认)、`densenet121`
  - 对应 `torchvision.models.resnet50` / `densenet121` 预训练权重，最后一层替换为特征输出。

- **多模态融合方式（`--fusion`）**
  - 可选值：`concat`、`weighted_sum`、`gated`、`attention`、`bilinear`
  - `concat`(默认)：特征拼接（基线方式）。
  - `weighted_sum`：学习标量权重，对文本/图像做加权求和。
  - `gated`：学习门控向量，对图像特征做缩放后与文本拼接。
  - `attention`：以文本为 query，对图像特征做简化注意力加权。
  - `bilinear`：文本/图像投影后逐元素相乘，建模细粒度交互。

- **消融模式（`--mode`）**：控制使用的模态
  - 可选值：`multimodal`、`text_only`、`image_only`
  - `multimodal`(默认)：默认多模态（文本 + 图像 特征）；
  - `text_only`：仅使用 BERT 文本分支；
  - `image_only`：仅使用图像分支（ResNet50 或 DenseNet121）。

训练完成后，每次运行会在 `outputs/run_YYYYMMDD_HHMMSS/` 下生成：

- `checkpoints/best.pt`：最优模型参数；
- `plots/loss_curve.png`、`plots/confusion_matrix.png`：训练曲线与验证集混淆矩阵；
- `model_params.json`：总参数量与可训练参数量；
- `summary.json`：最佳 epoch、各类指标与 loss 历史；
- `logs/epoch_*.json`：每个 epoch 的详细日志。

### 4.2 独立评估：`src.evaluate`

对已训练好的模型在训练集划分出的验证集上做独立评估：

```bash
python -m src.evaluate \
  --data_root project5 \
  --ckpt outputs/run_YYYYMMDD_HHMMSS/checkpoints/best.pt \
  --batch_size 16 \
  --max_length 128
```

脚本会加载与训练时一致的文本/图像预处理与模型结构，输出 Accuracy、Macro Precision、Macro Recall、Macro F1，并生成混淆矩阵图像。

### 4.3 测试集预测：`src.predict`

使用最优模型对 `test_without_label.txt` 进行预测，生成提交文件：

```bash
python -m src.predict \
  --data_root project5 \
  --ckpt outputs/run_YYYYMMDD_HHMMSS/checkpoints/best.pt \
  --fusion concat \
  --image_backbone densenet121 \
  --batch_size 16 \
  --max_length 128 \
  --output_path outputs/run_YYYYMMDD_HHMMSS/test_with_pred.txt
```

其中 `--fusion` 与 `--image_backbone` 需与训练该模型时保持一致。输出文件格式为：

```text
guid,tag
4597,negative
...
```

---

## 5. 开源仓库与参考文献

### 5.1 开源仓库

- **Joint Fine-Tuning for Multimodal Sentiment Analysis**  
  - 仓库：https://github.com/guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis
  - 参考点：多模态联合微调策略、参数初始化与训练流程设计。

- **Self-Attention vs Cross-Attention for Emotion Recognition**  
  - 仓库：https://github.com/smartcameras/SelfCrossAttn 
  - 参考点：自注意力与跨注意力在多模态情感/表情识别中的对比实验设计、多模态融合机制实现。


### 5.2 参考文献
- W. Liu et al.  
  *Multimodal Sentiment Analysis With Image-Text Interaction Network*.  
  IEEE Transactions on Multimedia, 2022.  

- Jacob Devlin et al.  
  *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.  
  NAACL-HLT, 2019.

- Kaiming He et al.
  *Deep Residual Learning for Image Recognition*.
  CVPR, 2016.