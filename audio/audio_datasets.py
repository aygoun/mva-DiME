"""
Audio datasets that return 3-channel mel-spectrogram tensors in [-1, 1],
compatible with the image-based DDPM from the original DiME pipeline.
"""

import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from .spectrogram_utils import (
    audio_to_tensor, load_audio, audio_to_mel, spec_to_tensor,
    spec_augment, random_time_shift,
    SAMPLE_RATE, SPEC_SIZE,
)


NUM_AUDIOSET_CLASSES = 527


class FSD50KDataset(Dataset):
    """FSD50K multi-label sound-event dataset, served as mel-spectrogram tensors.

    Each sample may carry **multiple** AudioSet labels.  Labels are returned as
    a 527-dim multi-hot vector using AudioSet class indices so that the
    pretrained AST model can be used directly without a custom head.

    Expected folder layout (standard FSD50K release)::

        data_dir/
            FSD50K.dev_audio/       # WAV files for development set
            FSD50K.eval_audio/      # WAV files for evaluation set
            FSD50K.ground_truth/
                dev.csv             # fname,labels,mids,split
                eval.csv            # fname,labels,mids
                vocabulary.csv      # (index,label,mid) or headerless

    Parameters
    ----------
    data_dir : str
        Root of the extracted FSD50K dataset.
    split : str
        One of ``"train"``, ``"val"`` (subsets of dev), ``"dev"`` (all dev),
        or ``"eval"``.
    audioset_mid_to_idx : dict
        Mapping ``{AudioSet_MID: model_output_index}`` (527 entries).
        Obtain via :func:`audio.audio_classifier.load_audioset_class_mapping`.
    """

    def __init__(self, data_dir, split="eval", sr=SAMPLE_RATE,
                 duration=7.0, size=SPEC_SIZE, augment=False,
                 audioset_mid_to_idx=None):
        gt_dir = os.path.join(data_dir, "FSD50K.ground_truth")

        # --- vocabulary ---
        vocab = self._read_vocabulary(os.path.join(gt_dir, "vocabulary.csv"))
        self.fsd50k_labels = vocab["label"].tolist()
        self.fsd50k_mids = vocab["mid"].tolist()
        self.num_fsd50k_classes = len(self.fsd50k_labels)

        self.audioset_mid_to_idx = audioset_mid_to_idx or {}

        self.valid_audioset_indices = sorted(
            {self.audioset_mid_to_idx[m] for m in self.fsd50k_mids
             if m in self.audioset_mid_to_idx}
        )

        # --- ground-truth split ---
        if split in ("train", "val"):
            df = pd.read_csv(os.path.join(gt_dir, "dev.csv"))
            df = df[df["split"] == split]
            audio_dir = os.path.join(data_dir, "FSD50K.dev_audio")
        elif split == "dev":
            df = pd.read_csv(os.path.join(gt_dir, "dev.csv"))
            audio_dir = os.path.join(data_dir, "FSD50K.dev_audio")
        elif split == "eval":
            df = pd.read_csv(os.path.join(gt_dir, "eval.csv"))
            audio_dir = os.path.join(data_dir, "FSD50K.eval_audio")
        else:
            raise ValueError(f"Unknown split: {split}")

        df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.fnames = df["fname"].astype(str).tolist()

        # --- parse multi-label annotations (using MIDs) ---
        self.audioset_indices = []
        for _, row in df.iterrows():
            mids = [m.strip() for m in str(row["mids"]).split(",")]
            as_idxs = [self.audioset_mid_to_idx[m] for m in mids
                       if m in self.audioset_mid_to_idx]
            self.audioset_indices.append(as_idxs)

        self.sr = sr
        self.duration = duration
        self.size = size
        self.augment = augment
        self.num_classes = NUM_AUDIOSET_CLASSES

    @staticmethod
    def _read_vocabulary(path):
        """Read vocabulary.csv, handling both with-header and headerless."""
        df = pd.read_csv(path)
        if {"label", "mid"}.issubset(df.columns):
            return df
        df = pd.read_csv(path, header=None, names=["index", "label", "mid"])
        return df

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.audio_dir, f"{self.fnames[idx]}.wav")

        try:
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
        except Exception:
            # Corrupted / truncated WAV — return silence so the batch isn't lost
            import numpy as np
            tensor = spec_to_tensor(
                np.full((128, 128), -80.0, dtype=np.float32), size=self.size,
            )
            time_frames = self.size

        multi_hot = torch.zeros(NUM_AUDIOSET_CLASSES, dtype=torch.float32)
        for aidx in self.audioset_indices[idx]:
            multi_hot[aidx] = 1.0

        return tensor, multi_hot, time_frames


def infinite_audio_loader(dataset, batch_size, shuffle=True, num_workers=4):
    """Yield (batch, cond_dict) pairs endlessly, matching TrainLoop's API."""
    use_pin = torch.cuda.is_available()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, pin_memory=use_pin,
    )
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                yield batch[0], {}
            else:
                yield batch, {}
