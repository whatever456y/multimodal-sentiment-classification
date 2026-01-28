from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from .datasets import ID2LABEL, LABEL2ID, MultiModalDataset, Sample, load_csv
from .models.baseline import ModelConfig, MultiModalBaseline
from .utils import ensure_dir, get_text_model_path, save_json


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Path) -> None:
    # 绘制并保存混淆矩阵图像
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # 动态计算阈值，用于决定文本颜色
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[Dict[str, float], np.ndarray]:
    # 评估模型在给定数据加载器上的表现
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].cpu().numpy().tolist()

        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()

        all_y.extend(labels)
        all_pred.extend(pred)

    acc = accuracy_score(all_y, all_pred)
    # 计算宏平均精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)
    # 生成混淆矩阵
    cm = confusion_matrix(all_y, all_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    return metrics, cm


def collate_fn(tokenizer, batch, max_length: int):
    # 数据加载器的批处理函数，处理文本编码和图像堆叠
    texts = [x["text"] for x in batch]
    images = torch.stack([x["image"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values": images,
        "labels": labels,
    }


def main() -> None:
    """评估脚本主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="project5")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_dir = data_root / "data"

    import json

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    val_guids = set(split["val_guids"])

    rows = load_csv(data_root / "train.txt")
    samples: List[Sample] = []
    for guid, tag in rows:
        if guid in val_guids:
            samples.append(Sample(guid=guid, label=LABEL2ID[tag]))

    image_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),# 统一图像尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    # 初始化分词器
    text_model_path = get_text_model_path("google-bert/bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    # 创建数据集和数据加载器
    ds = MultiModalDataset(data_dir=data_dir, samples=samples, image_transform=image_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = MultiModalBaseline(ModelConfig()).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)

    metrics, cm = evaluate(model, loader, device)

    out_dir = ensure_dir(args.output_dir)
    save_json(metrics, out_dir / "metrics.json")
    labels = [ID2LABEL[i] for i in [0, 1, 2]]
    plot_confusion_matrix(cm, labels, out_dir / "confusion_matrix.png")

    print(metrics)


if __name__ == "__main__":
    main()
