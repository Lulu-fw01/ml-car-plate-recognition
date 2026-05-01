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
        self.label_smoothing = 0.1
        self.val_cer = CharErrorRate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _label_smoothing_loss(self, log_probs, targets):
        """Label smoothing loss для CTC"""
        # Используем стандартный CTC как baseline для smoothing
        baseline_loss = nn.CTCLoss(
            blank=self.blank_idx, reduction="mean", zero_infinity=True
        )

        T, B, C = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)

        targets_list = []
        target_lengths_list = []
        for text in targets:
            t = [self.alphabet.index(c) for c in text if c in self.alphabet]
            targets_list.extend(t)
            target_lengths_list.append(len(t))

        targets = torch.tensor(targets_list, dtype=torch.long, device=self.device)
        target_lengths = torch.tensor(
            target_lengths_list, dtype=torch.long, device=self.device
        )

        return baseline_loss(log_probs, targets, input_lengths, target_lengths)

    def training_step(self, batch, batch_idx):
        images, texts = batch
        logits = self.forward(images)
        log_probs = F.log_softmax(logits, dim=-1)
        T, B, _ = logits.shape

        input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)

        targets_list = []
        target_lengths_list = []
        for text in texts:
            t = [self.alphabet.index(c) for c in text if c in self.alphabet]
            targets_list.extend(t)
            target_lengths_list.append(len(t))

        targets = torch.tensor(targets_list, dtype=torch.long, device=self.device)
        target_lengths = torch.tensor(
            target_lengths_list, dtype=torch.long, device=self.device
        )

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        smooth_loss = self._label_smoothing_loss(log_probs, texts)
        loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        if batch_idx % 100 == 0:
            decoded = self.decode_greedy(log_probs)
            cer = jiwer.cer(list(texts), list(decoded))
            self.log("train_cer", cer, prog_bar=True, batch_size=len(texts))

        self.log("train_loss", loss, prog_bar=True, batch_size=len(texts))
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        logits = self.model(images)
        log_probs = F.log_softmax(logits, dim=-1)
        T, B, _ = logits.shape

        input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
        targets_list, target_lengths_list = [], []
        for text in texts:
            t = [self.alphabet.index(c) for c in text if c in self.alphabet]
            targets_list.extend(t)
            target_lengths_list.append(len(t))

        targets = torch.tensor(targets_list, dtype=torch.long, device=self.device)
        target_lengths = torch.tensor(
            target_lengths_list, dtype=torch.long, device=self.device
        )

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        decoded = self.decode_greedy(log_probs)
        cer = jiwer.cer(list(texts), list(decoded))

        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(texts))
        self.log("val_cer", cer, prog_bar=True, sync_dist=True, batch_size=len(texts))
        return {"val_cer": cer, "val_loss": loss}

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
            self.parameters(), lr=self.hparams.config.trainer.lr, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=self.hparams.config.trainer.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_cer",
                "interval": "epoch",
                "frequency": 1,
            },
        }
