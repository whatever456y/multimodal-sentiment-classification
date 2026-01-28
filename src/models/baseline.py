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
    """模型配置类，定义多模态模型的关键参数"""
    text_model_name: str = "google-bert/bert-base-uncased" # 文本编码器名称
    num_classes: int = 3 # 分类类别数（negative, neutral, positive）
    proj_dim: int = 256  # 投影层维度，统一文本和图像特征的维度
    dropout: float = 0.1 # dropout率
    fusion_type: str = "concat"  # 融合策略：concat, weighted_sum, gated, attention, bilinear
    image_backbone: str = "resnet50"   # 图像编码器骨干：resnet50, densenet121
    mode: str = "multimodal"  # 运行模式：multimodal, text_only, image_only


class MultiModalBaseline(nn.Module):
    """多模态基线模型，支持多种融合策略"""
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ========== 文本编码器初始化 ==========
        if cfg.mode != "image_only":
            text_model_path = get_text_model_path(cfg.text_model_name)
            self.text_encoder = AutoModel.from_pretrained(text_model_path)
            text_hidden = self.text_encoder.config.hidden_size
            self.text_proj = nn.Linear(text_hidden, cfg.proj_dim)

        # ========== 图像编码器初始化 ==========
        if cfg.mode != "text_only":
            if cfg.image_backbone == "densenet121":
                weights = torchvision.models.DenseNet121_Weights.DEFAULT
                self.image_encoder = torchvision.models.densenet121(weights=weights)
                image_hidden = self.image_encoder.classifier.in_features
                self.image_encoder.classifier = nn.Identity()
            else:
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                self.image_encoder = torchvision.models.resnet50(weights=weights)
                image_hidden = self.image_encoder.fc.in_features
                self.image_encoder.fc = nn.Identity()
            self.image_proj = nn.Linear(image_hidden, cfg.proj_dim)

        # ========== 融合策略相关层初始化 ==========
        fusion_type = cfg.fusion_type
        if cfg.mode in {"text_only", "image_only"}:
            fusion_dim = cfg.proj_dim
        else:
            if fusion_type == "weighted_sum":
                # 加权求和融合:学习文本特征的权重α，加权融合两个模态
                self.alpha_layer = nn.Linear(cfg.proj_dim * 2, 1)
                fusion_dim = cfg.proj_dim
            elif fusion_type == "gated":
                # 门控融合:学习门控向量，对图像特征进行缩放
                self.gate_layer = nn.Linear(cfg.proj_dim * 2, cfg.proj_dim)
                fusion_dim = cfg.proj_dim * 2
            elif fusion_type == "attention":
                # 注意力融合:文本作为query，图像作为key-value
                self.att_q = nn.Linear(cfg.proj_dim, cfg.proj_dim)
                self.att_k = nn.Linear(cfg.proj_dim, cfg.proj_dim)
                self.att_v = nn.Linear(cfg.proj_dim, cfg.proj_dim)
                fusion_dim = cfg.proj_dim * 2
            elif fusion_type == "bilinear":
                # 双线性交互融合：特征投影后逐元素相乘
                self.bilinear_t = nn.Linear(cfg.proj_dim, cfg.proj_dim)
                self.bilinear_v = nn.Linear(cfg.proj_dim, cfg.proj_dim)
                fusion_dim = cfg.proj_dim
            else:
                fusion_dim = cfg.proj_dim * 2

        # ========== 分类器初始化 ==========
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
        # 前向传播
        mode = self.cfg.mode

        # ========== 文本特征提取 ==========
        if mode != "image_only":
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = text_out.last_hidden_state[:, 0]
            t = self.text_proj(cls)
        else:
            t = None
        
        # ========== 图像特征提取 ==========
        if mode != "text_only":
            v = self.image_encoder(pixel_values)
            v = self.image_proj(v)
        else:
            v = None

        # ========== 特征融合 ==========
        if mode == "text_only":
            z = t
        elif mode == "image_only":
            z = v
        else:
            fusion_type = self.cfg.fusion_type
            if fusion_type == "weighted_sum":
                h = torch.cat([t, v], dim=-1)
                alpha = torch.sigmoid(self.alpha_layer(h))
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
        
        # ========== 分类预测 ==========
        logits = self.classifier(z)
        return logits
