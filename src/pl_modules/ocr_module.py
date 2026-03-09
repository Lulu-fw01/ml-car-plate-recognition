import pytorch_lightning as pl
import torch
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate
from models.crnn import CRNN
import torch.nn.functional as F


class OCRModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        alphabet = config.model.alphabet
        self.model = CRNN(
            img_h=config.model.img_h,
            nc=1,
            nclass=len(alphabet),
            nh=config.model.rnn_hidden,
            n_rnn=config.model.rnn_layers,
        )
        self.criterion = CTCLoss(blank=len(alphabet), zero_infinity=True)  # todo check
        self.val_cer = CharErrorRate()

    def training_step(self, batch, batch_idx):
        images, targets = batch  # targets: (B, 9) — индексы символов
        logits = self.model(images)  # (B, 9, 22)
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets, ignore_index=-1)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, input_lengths, target_lengths = batch
        logits = self.model(images)
        log_probs = torch.log_softmax(logits, dim=2)
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Greedy decode
        preds = log_probs.argmax(2).T  # (B, T)
        decoded = []
        for p in preds:
            # Удалить blank и повторы
            prev = -1
            seq = []
            for c in p:
                if c != prev and c != len(self.hparams.config.model.crnn.alphabet):
                    seq.append(c.item())
                prev = c
            decoded.append(seq)

        # Convert targets back to list of lists
        targets_list = []
        start = 0
        for length in target_lengths:
            targets_list.append(targets[start : start + length].tolist())
            start += length

        self.val_cer(decoded, targets_list)
        self.log("val_cer", self.val_cer, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
