from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from .datasets import LABEL2ID, MultiModalDataset, Sample, load_csv
from .models.baseline import ModelConfig, MultiModalBaseline
from .utils import AverageMeter, count_parameters, ensure_dir, get_text_model_path, save_json, set_seed


def collate_fn(tokenizer, batch, max_length: int, text_transform=None, train: bool = False):
    # 数据批处理函数，整合文本编码和图像处理
    texts = [x["text"] for x in batch]
    if text_transform is not None:
        texts = [text_transform(t, train=train) for t in texts]

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


_url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_user_pattern = re.compile(r"@\w+")
_multi_punct_pattern = re.compile(r"([!?]){2,}")


def clean_text(text: str) -> str:
    # 社交媒体文本清洗：标准化URL、用户提及和重复标点
    text = _url_pattern.sub("<URL>", text)
    text = _user_pattern.sub("<USER>", text)
    text = _multi_punct_pattern.sub(r"\1", text)
    return text.strip()


def simple_augment(text: str) -> str:
    # 轻量级文本增强：随机删除非关键token，避免破坏情感极性

    tokens = text.split()
    if len(tokens) <= 3:
        return text
    # 随机删除 1 个 token（概率较低）
    if random.random() < 0.5:
        idx = random.randint(0, len(tokens) - 1)
        del tokens[idx]
    return " ".join(tokens)


def make_text_transform(config: str):
    # 根据配置创建文本处理函数

    def transform(text: str, train: bool = False) -> str:
        if config in {"clean", "aug", "all"}:
            text_out = clean_text(text)
        else:
            text_out = text

        if config in {"aug", "all"} and train:
            text_out = simple_augment(text_out)
        return text_out

    return transform


@torch.no_grad()
def evaluate(model, loader, device, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
    # 模型评估函数
    model.eval()
    loss_meter = AverageMeter()
    all_y: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), n=labels.size(0))

        pred = logits.argmax(dim=-1)
        all_y.extend(labels.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(all_y, all_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    return loss_meter.avg, metrics


def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: Path) -> None:
    # 绘制训练-验证损失曲线
    fig = plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: Path) -> None:
    # 绘制混淆矩阵
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def predict_on_loader(model, loader, device) -> Tuple[List[int], List[int]]:
    # 批量预测获取真实标签和预测标签
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        pred = logits.argmax(dim=-1)

        all_y.extend(labels.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    return all_y, all_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="project5")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resnet_lr_scale", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--fusion",
        type=str,
        default="concat",
        choices=["concat", "weighted_sum", "gated", "attention", "bilinear"],
        help="Fusion strategy for text and image features.",
    )
    parser.add_argument(
        "--image_backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "densenet121"],
        help="Backbone for image encoder.",
    )
    parser.add_argument(
        "--image_aug",
        type=str,
        default="base",
        choices=["base", "light", "strong"],
        help="Image augmentation config for training images.",
    )
    parser.add_argument(
        "--text_config",
        type=str,
        default="base",
        choices=["base", "clean", "aug", "twitter", "all"],
        help="Text configuration: preprocessing / augmentation / encoder choice.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_only", "image_only"],
        help="Training mode: multimodal / text_only / image_only.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(args.output_root) / run_name)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    plot_dir = ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "logs")

    data_root = Path(args.data_root)
    data_dir = data_root / "data"

    rows = load_csv(data_root / "train.txt")
    guids = [g for g, _ in rows]
    labels_str = [t for _, t in rows]
    labels = [LABEL2ID[t] for t in labels_str]

    train_guids, val_guids, train_y, val_y = train_test_split(
        guids,
        labels,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels,
    )

    save_json(
        {
            "seed": args.seed,
            "val_size": args.val_size,
            "train_guids": train_guids,
            "val_guids": val_guids,
        },
        out_dir / "split.json",
    )

    train_samples = [Sample(guid=g, label=y) for g, y in zip(train_guids, train_y)]
    val_samples = [Sample(guid=g, label=y) for g, y in zip(val_guids, val_y)]

    # 图像增强：根据 image_aug 选择不同强度的训练增强策略
    if args.mode == "text_only":
        # 文本单模态时仍保留一个固定的占位图像张量形状（不会被使用）
        image_tf_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif args.image_aug == "light":
        # 轻度增强：随机裁剪+翻转+颜色抖动
        image_tf_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif args.image_aug == "strong":
        # 强增强：更大范围的扰动
        image_tf_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # base：仅做基础预处理，不进行随机增强
        image_tf_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    # 验证集始终使用基础预处理，保持可比性
    image_tf_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 选择文本编码模型：仅 twitter/all 使用 Twitter 专用模型，其余配置使用基线 BERT
    if args.mode == "image_only":
        # 图像单模态时仍需要一个 tokenizer 来生成占位 input_ids
        text_model_name = "google-bert/bert-base-uncased"
    else:
        if args.text_config in {"twitter", "all"}:
            text_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        else:
            text_model_name = "google-bert/bert-base-uncased"

    text_model_path = get_text_model_path(text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    text_transform = make_text_transform(args.text_config)

    # ========== 创建数据集和数据加载器 ==========
    ds_train = MultiModalDataset(data_dir=data_dir, samples=train_samples, image_transform=image_tf_train)
    ds_val = MultiModalDataset(data_dir=data_dir, samples=val_samples, image_transform=image_tf_val)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length, text_transform=text_transform, train=True),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length, text_transform=text_transform, train=False),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(
        fusion_type=args.fusion,
        text_model_name=text_model_name,
        image_backbone=args.image_backbone,
        mode=args.mode,
    )
    model = MultiModalBaseline(model_cfg).to(device)

    # ========== 参数冻结策略（迁移学习） ==========
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is not None and args.mode != "image_only":
        embeddings = getattr(text_encoder, "embeddings", None)
        if embeddings is not None:
            for p in embeddings.parameters():
                p.requires_grad = False

        encoder = getattr(text_encoder, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            layers = encoder.layer
            for layer in layers[:10]:
                for p in layer.parameters():
                    p.requires_grad = False

    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is not None and args.mode != "text_only":
        for name in ["conv1", "bn1", "layer1", "layer2", "layer3"]:
            module = getattr(image_encoder, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

    total_params, trainable_params = count_parameters(model)
    save_json(
        {"total_params": int(total_params), "trainable_params": int(trainable_params)},
        out_dir / "model_params.json",
    )

    # ========== 损失函数和优化器配置 ==========
    criterion = nn.CrossEntropyLoss()

    resnet_group_params: List[torch.nn.Parameter] = []
    if image_encoder is not None and args.mode != "text_only":
        if hasattr(image_encoder, "layer4"):
            resnet_group_params.extend(list(image_encoder.layer4.parameters()))
    if hasattr(model, "image_proj") and args.mode != "text_only":
        resnet_group_params.extend(list(model.image_proj.parameters()))

    resnet_param_ids = {id(p) for p in resnet_group_params}
    all_trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_group_params = [p for p in all_trainable_params if id(p) not in resnet_param_ids]

    optimizer_group_params = []
    if base_group_params:
        optimizer_group_params.append({"params": base_group_params})
    if resnet_group_params:
        optimizer_group_params.append({"params": resnet_group_params, "lr": args.lr * args.resnet_lr_scale})

    optimizer = torch.optim.AdamW(optimizer_group_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_f1 = -1.0
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        for batch in dl_train:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels_t = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            loss = criterion(logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_meter.update(loss.item(), n=labels_t.size(0))

        train_loss = loss_meter.avg
        val_loss, val_metrics = evaluate(model, dl_val, device, criterion)

        scheduler.step()

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        print(epoch_log)

        save_json(epoch_log, out_dir / "logs" / f"epoch_{epoch}.json")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            best_metrics = {k: float(v) for k, v in val_metrics.items()}
            no_improve_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.early_stop_patience:
            break

    # ========== 训练后处理 ==========
    plot_loss_curve(train_losses, val_losses, plot_dir / "loss_curve.png")

    best_state = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_state["model"], strict=True)
    y_true, y_pred = predict_on_loader(model, dl_val, device)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plot_confusion_matrix(cm, labels=["negative", "neutral", "positive"], save_path=plot_dir / "confusion_matrix.png")

    save_json(
        {
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_f1),
            "best_val_metrics": best_metrics,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        out_dir / "summary.json",
    )

    print(f"Best epoch={best_epoch}, best val f1_macro={best_f1:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
