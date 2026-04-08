"""PANNs CNN14-based perceptual loss for 1-channel mel spectrograms.

Replaces the VGG19 perceptual loss with an audio-native feature extractor.
CNN14 was trained on AudioSet (32 kHz, 64 mel bins) — the same domain as our
classifier — so its intermediate features capture acoustically meaningful
patterns (pitch, timbre, onsets) rather than visual textures.

Weights are auto-downloaded from Zenodo on first use (~320 MB).
"""

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

_WEIGHTS_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "panns")
_WEIGHTS_FILE = os.path.join(_CACHE_DIR, "Cnn14_mAP=0.431.pth")


def _download_weights() -> str:
    if os.path.isfile(_WEIGHTS_FILE):
        return _WEIGHTS_FILE
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"Downloading CNN14 weights → {_WEIGHTS_FILE} ...")
    torch.hub.download_url_to_file(_WEIGHTS_URL, _WEIGHTS_FILE)
    return _WEIGHTS_FILE


class _ConvBlock(nn.Module):
    """Two-layer conv block matching PANNs' ConvBlock (no dropout)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


# Pool sizes used in the original CNN14 forward pass
_POOL_SIZES = [(2, 2)] * 5 + [(1, 1)]

# Channel dims per block
_CHANNELS = [
    (1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048),
]


class CNN14PerceptualLoss(nn.Module):
    """Perceptual loss using PANNs CNN14 intermediate features.

    Natively operates on **1-channel** mel spectrograms — no 3-channel hack.

    Input tensors should be ``(B, 1, 256, 256)`` in ``[-1, 1]``, matching
    the audio-diffusion DDPM output format.

    Internally the spectrograms are resized from 256 mel bins to 64 to match
    CNN14's pretrained statistics, and re-arranged to ``(B, 1, time, mel)``
    before feature extraction.

    Parameters
    ----------
    layer : int
        Number of CNN14 conv blocks to use for feature extraction (1–6).
        Higher = deeper / more abstract features.  Default 4.
    c : float
        Scalar coefficient for the loss.
    """

    def __init__(self, layer: int = 4, c: float = 1.0):
        super().__init__()
        assert 1 <= layer <= 6, f"layer must be 1-6, got {layer}"
        self.c = c
        self.n_blocks = layer

        self.bn0 = nn.BatchNorm2d(64)
        self.blocks = nn.ModuleList(
            [_ConvBlock(*_CHANNELS[i]) for i in range(layer)]
        )

        ckpt = _download_weights()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        self._load_partial(state)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ------------------------------------------------------------------
    def _load_partial(self, state_dict: dict):
        my_state = self.state_dict()
        mapped = {}

        for suffix in ("weight", "bias", "running_mean",
                        "running_var", "num_batches_tracked"):
            key = f"bn0.{suffix}"
            if key in state_dict and key in my_state:
                mapped[key] = state_dict[key]

        for i in range(self.n_blocks):
            src = f"conv_block{i + 1}."
            dst = f"blocks.{i}."
            for k, v in state_dict.items():
                if k.startswith(src):
                    new_key = dst + k[len(src):]
                    if new_key in my_state:
                        mapped[new_key] = v

        self.load_state_dict(mapped, strict=False)
        n_loaded = len(mapped)
        n_total = len(my_state)
        print(f"CNN14PerceptualLoss: loaded {n_loaded}/{n_total} params "
              f"(blocks 1-{self.n_blocks})")

    # ------------------------------------------------------------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, freq=256, time=256) [-1,1]`` → ``(B, 1, time=256, mel=64)``."""
        x = x.transpose(2, 3)
        x = F.interpolate(x, size=(256, 64), mode="bilinear",
                          align_corners=False)
        return x

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)

        x = x.transpose(1, 3)          # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)          # (B, 1, T, 64)

        for i, block in enumerate(self.blocks):
            x = block(x, pool_size=_POOL_SIZES[i])
        return x

    # ------------------------------------------------------------------
    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        f0 = self._features(x0).reshape(B, -1)
        f1 = self._features(x1).reshape(B, -1)
        loss = F.mse_loss(f0, f1, reduction="none").mean(dim=1)
        return self.c * loss.sum()

    def train(self, mode: bool = True):
        return super().train(False)
