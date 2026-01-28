from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from .datasets import LABEL2ID, ID2LABEL, MultiModalDataset, Sample, load_csv
from .models.baseline import ModelConfig, MultiModalBaseline
from .utils import get_text_model_path, set_seed


def build_val_dataset(data_root: Path, val_size: float, seed: int) -> MultiModalDataset:
    """构建验证集数据集对象
    
    从完整训练集中划分出一部分作为验证集，用于错误案例分析。
    使用分层抽样确保各类别比例与原始数据集一致。
    
    Args:
        data_root: 数据根目录，包含data文件夹和train.txt
        val_size: 验证集比例，如0.2表示20%数据作为验证集
        seed: 随机种子，确保划分可复现
        
    Returns:
        验证集数据集对象，包含图像预处理流水线
    """
    data_dir = data_root / "data"
    rows = load_csv(data_root / "train.txt")
    guids = [g for g, _ in rows]
    labels_str = [t for _, t in rows]
    labels = [LABEL2ID[t] for t in labels_str]

    train_guids, val_guids, train_y, val_y = train_test_split(
        guids,
        labels,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )

    val_samples = [Sample(guid=g, label=y) for g, y in zip(val_guids, val_y)]

    image_tf_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_val = MultiModalDataset(data_dir=data_dir, samples=val_samples, image_transform=image_tf_val)
    return ds_val


def find_bad_cases(
    model: torch.nn.Module,
    tokenizer,
    ds_val: MultiModalDataset,
    device: torch.device,
    num_cases: int,
    data_root: Path,
) -> List[dict]:
    # 查找并收集模型误分类的样本
    loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    bad_cases: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            guid = batch["guid"][0]
            text = batch["text"][0]
            image = batch["image"][0].to(device).unsqueeze(0)
            label = int(batch["label"][0])

            enc = tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image)
            pred = int(logits.argmax(dim=-1).item())

            if pred != label:
                bad_cases.append(
                    {
                        "guid": guid,
                        "text": text,
                        "true_label": ID2LABEL[label],
                        "pred_label": ID2LABEL[pred],
                        "image_path": str((data_root / "data" / f"{guid}.jpg").resolve()),
                    }
                )

            if len(bad_cases) >= num_cases:
                break

    return bad_cases


def main() -> None:
    """主函数：加载训练好的模型，分析验证集上的错误样本"""
    parser = argparse.ArgumentParser(description="Show a few misclassified validation samples (bad cases).")
    parser.add_argument("--data_root", type=str, default="project5")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (e.g. outputs/.../best.pt)")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "weighted_sum", "gated", "attention", "bilinear"])
    parser.add_argument("--num_cases", type=int, default=5, help="Number of bad cases to display")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    ds_val = build_val_dataset(data_root, val_size=args.val_size, seed=args.seed)

    text_model_path = get_text_model_path("google-bert/bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(fusion_type=args.fusion)
    model = MultiModalBaseline(model_cfg).to(device)

    ckpt_path = Path(args.ckpt)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)

    bad_cases = find_bad_cases(model, tokenizer, ds_val, device, args.num_cases, data_root)

    if not bad_cases:
        print("No misclassified samples found with current settings.")
        return

    print("=== Bad case examples ===")
    for i, c in enumerate(bad_cases, start=1):
        print(f"\n[Case {i}]")
        print(f"GUID       : {c['guid']}")
        print(f"True label : {c['true_label']}")
        print(f"Pred label : {c['pred_label']}")
        print(f"Text       : {c['text']}")
        print(f"Image path : {c['image_path']}")


if __name__ == "__main__":
    main()
