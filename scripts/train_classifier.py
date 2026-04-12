from pathlib import Path

import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dense_audio_classifier.data.mel_irmas_datamodule import IRMASDataModule
from dense_audio_classifier.model.dense_classifier import DenseAudioClassifier

if __name__ == "__main__":
    # checkpoint local best
    model_checkpoint = ModelCheckpoint(
        monitor="train_acc_epoch",
        dirpath="checkpoints",
        mode="max",
        filename="dense_classifier_irmas-{epoch:02d}-{train_acc_epoch:.2f}",
        save_top_k=1,
        save_last=True,
    )
    wandb_logger = WandbLogger(
        project="xai-dime",
        log_model=True,
        name="train_dense_classifier_irmas",
        checkpoint_name="train_dense_classifier_irmas",
    )

    trainer = Trainer(
        accelerator="cuda",
        max_epochs=100,
        callbacks=[model_checkpoint],
        logger=wandb_logger,
        devices=1,
        num_nodes=1,
    )
    model = DenseAudioClassifier()
    datamodule = IRMASDataModule(
        batch_size=32,
        num_workers=4,
        base_data=Path("notebooks/data"),
    )

    trainer.fit(model, datamodule=datamodule)

    wandb.finish()
