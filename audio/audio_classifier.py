"""
Audio classifier backbones for DiME counterfactual guidance.

Supports:
- ResNet18 on spectrogram "images"
- AST backbone from Hugging Face with an ESC-50 head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from huggingface_hub import snapshot_download


class SpectrogramClassifier(nn.Module):
    """ResNet18-based classifier over 3-channel mel-spectrogram images."""

    def __init__(self, num_classes, weights_path=None):
        super().__init__()

        weights = torchvision.models.ResNet18_Weights.DEFAULT if weights_path is None else None
        backbone = torchvision.models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone = backbone
        self.num_classes = num_classes

        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("sigma", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state)

    def forward(self, x):
        x = (torch.clamp(x, -1, 1) + 1) / 2
        x = (x - self.mu) / self.sigma
        return self.backbone(x)


class ASTESC50Classifier(nn.Module):
    """
    AST backbone + custom ESC-50 head.

    Input:
      x in [-1, 1], shape (B, 3, H, W)
    Internally:
      - take first channel
      - convert to AST expected shape (B, time, mel_bins)
      - run AST backbone and classify with a trainable linear head
    """

    def __init__(self, num_classes, model_id="MIT/ast-finetuned-audioset-10-10-0.4593", weights_path=None):
        super().__init__()
        from transformers import ASTModel

        self.backbone = ASTModel.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True, # we can ignore because we add a custom head
        )
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes
        # This checkpoint expects 128 mel bins and 1024 time frames (10s setup).
        # We resize incoming spectrograms to this shape internally.
        self.expected_mel_bins = 128
        self.expected_time_frames = 1024

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state)

    def forward(self, x):
        # x: (B, 3, H, W) in [-1, 1]
        x = torch.clamp(x, -1, 1)
        x = x[:, :1, :, :]               # (B, 1, H, W)
        x = F.interpolate(
            x,
            size=(self.expected_mel_bins, self.expected_time_frames),
            mode="bilinear",
            align_corners=False,
        )
        x = x[:, 0, :, :]                # (B, mel_bins, time)
        x = x.transpose(1, 2)            # (B, time, mel_bins)
        outputs = self.backbone(input_values=x)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def ensure_hf_model_downloaded(model_id, local_dir="audio/models/hf"):
    """
    Download a Hugging Face model repository locally using the HF SDK.
    Returns the resolved local path.
    """
    path = snapshot_download(repo_id=model_id, local_dir=local_dir)
    return path


def build_classifier(classifier_type, num_classes, weights_path=None, ast_model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
    if classifier_type == "resnet18":
        return SpectrogramClassifier(num_classes=num_classes, weights_path=weights_path)
    if classifier_type == "ast":
        return ASTESC50Classifier(num_classes=num_classes, model_id=ast_model_id, weights_path=weights_path)
    raise ValueError(f"Unknown classifier_type: {classifier_type}")
