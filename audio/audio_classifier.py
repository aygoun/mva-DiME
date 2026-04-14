"""Audio classifier helpers for DiME's audio pipeline."""

from pathlib import Path

import torch
from torch import nn

from dense_audio_classifier.model.dense_classifier import DenseAudioClassifier


def _resolve_checkpoint_path(
    checkpoint_path: str | None,
    wandb_artifact: str | None,
    wandb_root: str = "audio/models/wandb",
) -> str:
    """Resolve a local checkpoint path, optionally downloading from W&B."""
    artifact_ref = wandb_artifact
    if artifact_ref is None and checkpoint_path and checkpoint_path.startswith("wandb://"):
        artifact_ref = checkpoint_path[len("wandb://") :]

    if artifact_ref is None:
        if checkpoint_path is None:
            raise ValueError("Either checkpoint_path or wandb_artifact must be provided.")
        return checkpoint_path

    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    download_dir = Path(wandb_root) / artifact_ref.replace("/", "__").replace(":", "__")
    artifact_dir = Path(artifact.download(root=str(download_dir)))
    ckpt_files = sorted(artifact_dir.rglob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt file found in downloaded artifact: {artifact_ref}"
        )
    return str(ckpt_files[0])


def _num_classes_from_checkpoint(path: str) -> int:
    """Infer num_classes from the classifier head stored in a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    weight = state["densenet.classifier.weight"]
    return weight.shape[0]


def build_classifier(
    checkpoint_path: str | None,
    num_classes: int | None = None,
    wandb_artifact: str | None = None,
) -> nn.Module:
    """Load DenseAudioClassifier from local path or W&B artifact.

    If *num_classes* is ``None`` (or 0), it is auto-detected from the
    checkpoint's classifier head so the model always matches the saved
    weights.
    """
    resolved_path = _resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        wandb_artifact=wandb_artifact,
    )
    if not num_classes:
        num_classes = _num_classes_from_checkpoint(resolved_path)
        print(f"  Auto-detected num_classes={num_classes} from checkpoint")
    model = DenseAudioClassifier.load_from_checkpoint(
        checkpoint_path=resolved_path,
        num_classes=num_classes,
    )
    return _AudioChannelAdapter(model)


class _AudioChannelAdapter(nn.Module):
    """Adapter that accepts 1-channel mel tensors for DenseAudioClassifier."""

    def __init__(self, model: DenseAudioClassifier):
        super().__init__()
        self.model = model

    @property
    def num_classes(self) -> int:
        return self.model.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
