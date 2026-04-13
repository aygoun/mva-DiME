import lightning as L
import torch
from torch import nn
from torchmetrics.classification import MultilabelAccuracy
from torchvision.models.densenet import densenet121  # type: ignore


class DenseAudioClassifier(L.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.densenet = densenet121()
        self.densenet.classifier = nn.Linear(in_features=1024, out_features=num_classes)

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.test_accuracy = MultilabelAccuracy(num_labels=num_classes)
        self.train_accuracy = MultilabelAccuracy(num_labels=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.densenet(x)
        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y, mask = batch["mel"], batch["label"].float(), batch["mask"].float()
        x = x.repeat(1, 3, 1, 1)
        logits = self(x)
        loss = (self.criterion(logits, y) * mask).sum() / mask.sum()
        self.train_accuracy(logits, (y > 0.5).int())
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_epoch", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y, mask = batch["mel"], batch["label"].float(), batch["mask"].float()
        x = x.repeat(1, 3, 1, 1)
        logits = self(x)
        loss = (self.criterion(logits, y) * mask).sum() / mask.sum()
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", self.test_accuracy, on_step=True, on_epoch=True)
        self.test_accuracy(logits, (y > 0.5).int())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
