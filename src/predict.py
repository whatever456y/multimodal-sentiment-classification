from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from .datasets import ID2LABEL, MultiModalDataset, Sample, load_csv
from .models.baseline import ModelConfig, MultiModalBaseline
from .utils import get_text_model_path


def collate_fn(tokenizer, batch, max_length: int):
    texts = [x["text"] for x in batch]
    images = torch.stack([x["image"] for x in batch])
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "guids": [x["guid"] for x in batch],
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values": images,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="project5")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_path", type=str, default="outputs/test_with_pred.txt")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_dir = data_root / "data"
    test_csv = data_root / "test_without_label.txt"

    rows = load_csv(test_csv)
    samples = [Sample(guid=g, label=None) for g, _ in rows]

    image_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    text_model_path = get_text_model_path("google-bert/bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    ds = MultiModalDataset(data_dir=data_dir, samples=samples, image_transform=image_tf)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalBaseline(ModelConfig()).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    pred_map = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()

        for guid, p in zip(batch["guids"], pred):
            pred_map[guid] = ID2LABEL[int(p)]

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep same format as input: guid,tag
    with out_path.open("w", encoding="utf-8") as f:
        f.write("guid,tag\n")
        for guid, _ in rows:
            f.write(f"{guid},{pred_map[guid]}\n")

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
