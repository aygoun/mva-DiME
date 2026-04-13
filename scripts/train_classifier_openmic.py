from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from dense_audio_classifier.data.mel_openmic_datamodule import (
    INSTRUMENTS,
    OpenMICDataModule,
)
from dense_audio_classifier.model.dense_classifier import DenseAudioClassifier

if __name__ == "__main__":
    # checkpoint local best
    model_checkpoint = ModelCheckpoint(
        monitor="train_acc_epoch",
        dirpath="checkpoints",
        mode="max",
        filename="dense_classifier_openmic-{epoch:02d}-{train_acc_epoch:.2f}",
        save_top_k=1,
        save_last=True,
    )
    wandb_logger = WandbLogger(
        project="xai-dime",
        log_model=True,
        name="train_dense_classifier_openmic",
        checkpoint_name="train_dense_classifier_openmic",
    )

    trainer = Trainer(
        accelerator="cuda",
        max_epochs=100,
        callbacks=[model_checkpoint],
        logger=wandb_logger,
        devices=1,
        num_nodes=1,
    )
    model = DenseAudioClassifier(num_classes=len(INSTRUMENTS))
    datamodule = OpenMICDataModule(
        batch_size=32,
        num_workers=4,
        num_workers_optimize=1,
        base_data=Path("notebooks/data"),
    )

    trainer.fit(model, datamodule=datamodule)

    wandb.finish()
