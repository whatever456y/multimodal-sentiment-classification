from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def _read_text_robust(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-8", errors="ignore").strip()


@dataclass
class Sample:
    guid: str
    label: Optional[int]


def load_csv(path: str | Path) -> List[Tuple[str, str]]:
    path = Path(path)
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            guid = str(row[0]).strip()
            tag = str(row[1]).strip() if len(row) > 1 else ""
            rows.append((guid, tag))
    return rows


class MultiModalDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        samples: List[Sample],
        image_transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.samples = samples
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        txt_path = self.data_dir / f"{s.guid}.txt"
        img_path = self.data_dir / f"{s.guid}.jpg"

        text = _read_text_robust(txt_path) if txt_path.exists() else ""

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.image_transform is not None:
            image = self.image_transform(image)

        item = {
            "guid": s.guid,
            "text": text,
            "image": image,
        }
        if s.label is not None:
            item["label"] = int(s.label)
        return item
