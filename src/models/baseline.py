from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel

from ..utils import get_text_model_path


@dataclass
class ModelConfig:
    text_model_name: str = "google-bert/bert-base-uncased"
    num_classes: int = 3
    proj_dim: int = 256
    dropout: float = 0.1
    fusion_type: str = "concat"  # concat, weighted_sum, gated, attention, bilinear


class MultiModalBaseline(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        text_model_path = get_text_model_path(cfg.text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_path)
        text_hidden = self.text_encoder.config.hidden_size

        weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.image_encoder = torchvision.models.resnet50(weights=weights)
        image_hidden = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        self.text_proj = nn.Linear(text_hidden, cfg.proj_dim)
        self.image_proj = nn.Linear(image_hidden, cfg.proj_dim)

        fusion_type = cfg.fusion_type
        if fusion_type == "weighted_sum":
            self.alpha_layer = nn.Linear(cfg.proj_dim * 2, 1)
            fusion_dim = cfg.proj_dim
        elif fusion_type == "gated":
            self.gate_layer = nn.Linear(cfg.proj_dim * 2, cfg.proj_dim)
            fusion_dim = cfg.proj_dim * 2
        elif fusion_type == "attention":
            self.att_q = nn.Linear(cfg.proj_dim, cfg.proj_dim)
            self.att_k = nn.Linear(cfg.proj_dim, cfg.proj_dim)
            self.att_v = nn.Linear(cfg.proj_dim, cfg.proj_dim)
            fusion_dim = cfg.proj_dim * 2
        elif fusion_type == "bilinear":
            self.bilinear_t = nn.Linear(cfg.proj_dim, cfg.proj_dim)
            self.bilinear_v = nn.Linear(cfg.proj_dim, cfg.proj_dim)
            fusion_dim = cfg.proj_dim
        else:
            fusion_dim = cfg.proj_dim * 2

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(fusion_dim, cfg.proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim, cfg.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa-style models may not have pooler_output
        cls = text_out.last_hidden_state[:, 0]
        t = self.text_proj(cls)

        v = self.image_encoder(pixel_values)
        v = self.image_proj(v)

        fusion_type = self.cfg.fusion_type
        if fusion_type == "weighted_sum":
            h = torch.cat([t, v], dim=-1)
            alpha = torch.sigmoid(self.alpha_layer(h))  # (B, 1)
            z = alpha * t + (1.0 - alpha) * v
        elif fusion_type == "gated":
            h = torch.cat([t, v], dim=-1)
            gate = torch.sigmoid(self.gate_layer(h))
            v_gated = gate * v
            z = torch.cat([t, v_gated], dim=-1)
        elif fusion_type == "attention":
            q = self.att_q(t)
            k = self.att_k(v)
            v_val = self.att_v(v)
            score = (q * k).sum(dim=-1, keepdim=True) / (q.size(-1) ** 0.5)
            alpha = F.softmax(score, dim=-1)
            v_att = alpha * v_val
            z = torch.cat([t, v_att], dim=-1)
        elif fusion_type == "bilinear":
            t_b = self.bilinear_t(t)
            v_b = self.bilinear_v(v)
            z = t_b * v_b
        else:
            z = torch.cat([t, v], dim=-1)

        logits = self.classifier(z)
        return logits
