import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MultilabelAccuracy
from torchvision.models.densenet import densenet121  # type: ignore

from dense_audio_classifier.data.mel_irmas_datamodule import INSTRUMENTS


class DenseAudioClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        num_classes = len(INSTRUMENTS)
        self.densenet = densenet121()
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=num_classes)

        self.criterion = nn.BCEWithLogitsLoss()
        self.test_accuracy = MultilabelAccuracy(num_labels=num_classes)
        self.train_accuracy = MultilabelAccuracy(num_labels=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.densenet(x)
        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch["mel"], batch["label"]
        y = y.float()
        x = x.repeat(1, 3, 1, 1)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_epoch", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x: torch.Tensor
        y: torch.Tensor
        x, y = batch["mel"], batch["label"]
        y = y.float()
        x = x.repeat(1, 3, 1, 1)
        logits = self(x)
        loss = self.criterion(logits, y)
        # class accuracy

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.test_accuracy, on_step=True, on_epoch=True)
        self.test_accuracy(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
