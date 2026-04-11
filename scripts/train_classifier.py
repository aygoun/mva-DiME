from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from dense_audio_classifier.data.mel_irmas_datamodule import IRMASDataModule

from dense_audio_classifier.model.dense_classifier import DenseAudioClassifier

# checkpoint local best
model_checkpoint = ModelCheckpoint(
    monitor="test_acc_epoch", dirpath="checkpoints", mode="max"
)
trainer = Trainer(accelerator="cuda", max_epochs=200, callbacks=[model_checkpoint])
model = DenseAudioClassifier()
datamodule = IRMASDataModule(batch_size=32)


trainer.fit(model, datamodule=datamodule)
