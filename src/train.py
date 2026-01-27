from __future__ import annotations

import argparse
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


def collate_fn(tokenizer, batch, max_length: int):
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


@torch.no_grad()
def evaluate(model, loader, device, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
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

    image_tf_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tf_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    text_model_path = get_text_model_path("google-bert/bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    ds_train = MultiModalDataset(data_dir=data_dir, samples=train_samples, image_transform=image_tf_train)
    ds_val = MultiModalDataset(data_dir=data_dir, samples=val_samples, image_transform=image_tf_val)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(fusion_type=args.fusion)
    model = MultiModalBaseline(model_cfg).to(device)

    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is not None:
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
    if image_encoder is not None:
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

    criterion = nn.CrossEntropyLoss()

    resnet_group_params: List[torch.nn.Parameter] = []
    if image_encoder is not None:
        if hasattr(image_encoder, "layer4"):
            resnet_group_params.extend(list(image_encoder.layer4.parameters()))
    if hasattr(model, "image_proj"):
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
