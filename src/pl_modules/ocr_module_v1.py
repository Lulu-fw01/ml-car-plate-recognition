import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from models.ocr_v1 import OCRv1


class OCRModuleV1(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        alphabet_len = len(config.model.alphabet) + 1
        self.model = OCRv1(
            num_chars=alphabet_len,
            max_length=config.model.max_length,
        )
        self.pad_idx = len(config.model.alphabet)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=alphabet_len, ignore_index=self.pad_idx
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=alphabet_len, ignore_index=self.pad_idx
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch  # targets: (B, 9)
        logits = self.model(images)  # (B, 9, 23)
        loss = F.cross_entropy(
            logits.permute(0, 2, 1),  # (B, 23, 9)
            targets,
            ignore_index=self.pad_idx,
        )
        preds = logits.argmax(dim=-1)  # (B, 9)
        self.train_acc(preds, targets)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.model(images)
        loss = F.cross_entropy(
            logits.permute(0, 2, 1), targets, ignore_index=self.pad_idx
        )
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, targets)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
