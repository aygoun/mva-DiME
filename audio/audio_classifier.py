"""
Audio classifier for DiME counterfactual guidance.

Uses the pretrained AST on AudioSet (527 multi-label classes).  FSD50K's 200
classes are a strict subset, so no custom head is needed.
"""

import os
import csv
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import ASTForAudioClassification

AUDIOSET_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/"
    "class_labels_indices.csv"
)
AUDIOSET_LABELS_CACHE = os.path.join(
    os.path.dirname(__file__), "data", "class_labels_indices.csv"
)


def load_audioset_class_mapping(cache_path=AUDIOSET_LABELS_CACHE):
    """Return ``{mid: int_index}`` for all 527 AudioSet classes.

    Downloads Google's ``class_labels_indices.csv`` on first call and caches
    it under ``audio/data/``.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.isfile(cache_path):
        print(f"Downloading AudioSet class labels -> {cache_path} ...")
        urllib.request.urlretrieve(AUDIOSET_LABELS_URL, cache_path)

    mid_to_idx = {}
    with open(cache_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            idx_str, mid, _display = row[0].strip(), row[1].strip(), row[2].strip()
            mid_to_idx[mid] = int(idx_str)

    return mid_to_idx


class ASTAudioSetClassifier(nn.Module):
    """Pretrained AST on AudioSet with the original 527-class head.

    Uses ``ASTForAudioClassification`` directly — no custom head is needed
    because FSD50K's 200 classes are a **subset** of AudioSet's 527.  During
    DiME guidance we simply index into the relevant output logit.

    Input:
      x in [-1, 1], shape (B, 3, H, W)
    Output:
      logits of shape (B, 527)
    """

    def __init__(self, model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(model_id)
        self.num_classes = self.model.config.num_labels
        self.expected_mel_bins = 128
        self.expected_time_frames = 1024

    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        x = x[:, :1, :, :]
        x = F.interpolate(
            x,
            size=(self.expected_mel_bins, self.expected_time_frames),
            mode="bilinear",
            align_corners=False,
        )
        x = x[:, 0, :, :]
        x = x.transpose(1, 2)
        outputs = self.model(input_values=x)
        return outputs.logits


def ensure_hf_model_downloaded(model_id, local_dir="audio/models/hf"):
    """Download a Hugging Face model repository locally using the HF SDK."""
    path = snapshot_download(repo_id=model_id, local_dir=local_dir)
    return path


def build_classifier(ast_model_id="MIT/ast-finetuned-audioset-10-10-0.4593"):
    """Build the pretrained AST AudioSet classifier (527 classes)."""
    return ASTAudioSetClassifier(model_id=ast_model_id)
