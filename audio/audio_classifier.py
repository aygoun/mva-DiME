"""Audio classifier helpers for DiME's audio pipeline."""

from pathlib import Path

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


def build_classifier(
    checkpoint_path: str | None,
    num_classes: int,
    wandb_artifact: str | None = None,
) -> DenseAudioClassifier:
    """Load DenseAudioClassifier from local path or W&B artifact."""
    resolved_path = _resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        wandb_artifact=wandb_artifact,
    )
    return DenseAudioClassifier.load_from_checkpoint(
        checkpoint_path=resolved_path,
        num_classes=num_classes,
    )
