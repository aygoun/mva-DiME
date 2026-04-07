"""
Audio datasets that return 3-channel mel-spectrogram tensors in [-1, 1],
compatible with the image-based DDPM from the original DiME pipeline.
"""

import os
import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from .spectrogram_utils import (
    audio_to_tensor, load_audio, audio_to_mel, spec_to_tensor,
    spec_augment, random_time_shift,
    SAMPLE_RATE, SPEC_SIZE,
)

class ESC50Dataset(Dataset):
    """ESC-50 environmental-sound dataset, served as mel-spectrogram tensors.

    Expected folder layout (unzipped from the ESC-50 GitHub release)::

        data_dir/
            audio/          # 2000 .wav files
            meta/
                esc50.csv   # metadata with columns: filename, fold, target, category, ...

    Parameters
    ----------
    data_dir : str
        Root of the extracted ESC-50 dataset.
    folds : list[int] or None
        Which folds (1-5) to include.  ``None`` → all folds.
    sr : int
        Target sample rate.
    duration : float
        Clip length in seconds (ESC-50 clips are 5 s).
    size : int
        Spatial resolution of the output tensor (height = width).
    """

    def __init__(self, data_dir, folds=None, sr=SAMPLE_RATE, duration=5.0,
                 size=SPEC_SIZE, augment=False):
        meta = pd.read_csv(os.path.join(data_dir, "meta", "esc50.csv"))
        if folds is not None:
            meta = meta[meta["fold"].isin(folds)]
        meta = meta.reset_index(drop=True)

        self.audio_dir = os.path.join(data_dir, "audio")
        self.filenames = meta["filename"].tolist()
        self.targets = meta["target"].tolist()
        self.categories = meta["category"].tolist()
        self.sr = sr
        self.duration = duration
        self.size = size
        self.augment = augment

        cats = sorted(set(self.categories))
        self.cat_to_idx = {c: i for i, c in enumerate(cats)}
        self.num_classes = len(cats)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, self.filenames[idx])

        if self.augment:
            y = load_audio(path, sr=self.sr, duration=self.duration)
            y = random_time_shift(y, sr=self.sr, max_shift_sec=0.4)
            log_mel = audio_to_mel(y, sr=self.sr)
            time_frames = log_mel.shape[1]
            tensor = spec_to_tensor(log_mel, size=self.size)
            tensor = spec_augment(tensor)
        else:
            tensor, time_frames = audio_to_tensor(
                path, sr=self.sr, duration=self.duration, size=self.size,
            )

        label = self.targets[idx]
        return tensor, label, time_frames


def infinite_audio_loader(dataset, batch_size, shuffle=True, num_workers=4):
    """Yield (batch, cond_dict) pairs endlessly, matching TrainLoop's API."""
    use_pin = torch.cuda.is_available()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, pin_memory=use_pin,
    )
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                yield batch[0], batch[1]
            else:
                yield batch[0], {}
