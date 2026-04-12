from pathlib import Path

import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dense_audio_classifier.data.mel_irmas_datamodule import IRMASDataModule
from dense_audio_classifier.model.dense_classifier import DenseAudioClassifier

# checkpoint local best
model_checkpoint = ModelCheckpoint(
    monitor="test_acc_epoch",
    dirpath="checkpoints",
    mode="max",
    filename="dense_classifier_irmas-{epoch:02d}-{test_acc_epoch:.2f}",
    save_top_k=1,
)
wandb_logger = WandbLogger(
    project="xai-dime",
    log_model=True,
    name="train_dense_classifier_irmas",
    checkpoint_name="train_dense_classifier_irmas",
)

trainer = Trainer(
    accelerator="cuda",
    max_epochs=200,
    callbacks=[model_checkpoint],
    logger=wandb_logger,
    devices=4,
    num_nodes=1,
)
model = DenseAudioClassifier()
datamodule = IRMASDataModule(
    batch_size=32,
    num_workers=10,
    data_optimize_output_dir=Path("/gpfs/workdir/sassis/data/"),
)


trainer.fit(model, datamodule=datamodule)

wandb.finish()
