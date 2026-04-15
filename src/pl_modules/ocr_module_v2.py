import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import CharErrorRate
import jiwer

from models.ocr_v2 import ResNet34CRNN


class OCRModuleV2(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNet34CRNN(
            num_classes=config.model.num_classes,
            hidden_size=config.model.hidden_size,
            num_lstm_layers=config.model.num_lstm_layers,
        )

        self.blank_idx = config.ocr.blank_idx
        self.alphabet = config.ocr.alphabet

        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_idx, reduction="mean", zero_infinity=True
        )
        self.val_cer = CharErrorRate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, texts = batch
        logits = self.model(images)
        log_probs = F.log_softmax(logits, dim=-1)
        T, B, _ = logits.shape

        input_lengths = torch.full((B,), T, dtype=torch.long)
        targets, target_lengths = [], []
        for text in texts:
            target = [self.alphabet.index(c) for c in text if c in self.alphabet]
            targets.extend(target)
            target_lengths.append(len(target))

        targets = torch.tensor(targets, dtype=torch.long, device=logits.device)
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=logits.device
        )

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        decoded = self.decode_greedy(log_probs)
        cer = jiwer.cer(texts, decoded)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_cer", cer, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        logits = self.model(images)
        log_probs = F.log_softmax(logits, dim=-1)
        decoded = self.decode_greedy(log_probs)
        cer = jiwer.cer(texts, decoded)

        self.log("val_cer", cer, prog_bar=True, sync_dist=True)
        return {"val_cer": cer}

    def decode_greedy(self, logits: torch.Tensor) -> list[str]:
        decoded_texts = []
        predictions = logits.argmax(dim=-1).permute(1, 0)

        for pred in predictions:
            chars = []
            prev = -1
            for idx in pred:
                idx = idx.item()
                if idx != prev and idx != self.blank_idx:
                    if idx < len(self.alphabet):
                        chars.append(self.alphabet[idx])
                prev = idx
            decoded_texts.append("".join(chars))

        return decoded_texts

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.config.trainer.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.config.trainer.max_epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.config.trainer.lr)
