"""Mel-spectrogram ↔ tensor ↔ audio conversion.

Uses the deprecated ``Mel`` class from ``diffusers`` — the exact same code
that was used to create the training data for ``teticio/audio-diffusion-breaks-256``.
This guarantees that our spectrogram encoding / decoding is bit-identical to
what the DDPM was trained on.

Pipeline
--------
PIL Image (uint8, 256×256)  ←→  tensor (float32, 1×256×256, [-1, 1])
                                                  ↕  (DDPM operates here)
audio (float32 waveform)    ←→  PIL Image          (via ``Mel``)
"""

import numpy as np
import torch
from PIL import Image
from diffusers.pipelines.deprecated.audio_diffusion import Mel

mel_converter = Mel()

SAMPLE_RATE = mel_converter.sr
SPEC_SIZE = mel_converter.x_res
N_MELS = mel_converter.n_mels
N_FFT = mel_converter.n_fft
HOP_LENGTH = mel_converter.hop_length
TOP_DB = mel_converter.top_db
SLICE_SAMPLES = mel_converter.slice_size


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image (grayscale uint8 256×256) → ``(1, 256, 256)`` tensor in [-1, 1]."""
    if image.mode != "L":
        image = image.convert("L")
    arr = np.array(image, dtype=np.float32)
    return torch.from_numpy(arr / 255.0 * 2.0 - 1.0).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """``(C, H, W)`` tensor in [-1, 1] → grayscale PIL Image."""
    arr = tensor[0].detach().cpu().numpy()
    arr = np.clip((arr + 1.0) / 2.0 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` tensor in [-1, 1] → waveform via ``Mel.image_to_audio``."""
    return mel_converter.image_to_audio(tensor_to_image(tensor))


def audio_file_to_tensor(path: str, audio_slice: int = 0) -> torch.Tensor:
    """Load an audio file and return ``(1, 256, 256)`` tensor in [-1, 1]."""
    mel_converter.load_audio(path)
    image = mel_converter.audio_slice_to_image(audio_slice)
    return image_to_tensor(image)
