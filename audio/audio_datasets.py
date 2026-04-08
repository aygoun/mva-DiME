"""
Dataset loader for teticio/audio-diffusion-breaks-256.

Each sample is a pre-computed 256×256 grayscale mel spectrogram (PIL Image)
of a ~5 s breakbeat music slice.  No labels are stored — the AST classifier
provides pseudo-labels at inference time.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_AUDIOSET_CLASSES = 527


class AudioDiffusionBreaksDataset(Dataset):
    """HuggingFace dataset ``teticio/audio-diffusion-breaks-256``.

    Parameters
    ----------
    repo_id : str
        HuggingFace dataset repository.
    split : str
        Dataset split (the repo only has ``"train"``).
    max_samples : int or None
        Cap the number of samples (useful for quick tests).
    """

    def __init__(
        self,
        repo_id="teticio/audio-diffusion-breaks-256",
        split="train",
        max_samples=None,
    ):
        from datasets import load_dataset

        print(f"Loading dataset {repo_id} (split={split}) ...")
        self.ds = load_dataset(repo_id, split=split)
        if max_samples is not None and max_samples < len(self.ds):
            self.ds = self.ds.select(range(max_samples))
        print(f"  {len(self.ds)} samples loaded")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        if image.mode != "L":
            image = image.convert("L")
        arr = np.array(image, dtype=np.float32)
        tensor = torch.from_numpy(arr / 255.0 * 2.0 - 1.0).unsqueeze(0)
        return tensor, idx
