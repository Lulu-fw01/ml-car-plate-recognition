import pytorch_lightning as pl
import torch
import torch.nn as nn
import jiwer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from models.ocr_v4 import PlateRecognizer


class OCRModuleV4(pl.LightningModule):
    def __init__(
        self,
        config=None,
        total_steps=None,
        alphabet=None,
        num_classes=25,
        hidden_dim=256,
        max_len=9,
        nhead=8,
        num_layers=3,
    ):
        super().__init__()

        if config is not None:
            self.save_hyperparameters()
            self.total_steps = total_steps
            self.tf_ratio = 1.0

            self.char2idx = {c: i for i, c in enumerate(config.ocr.alphabet)}
            self.idx2char = {i: c for c, i in self.char2idx.items()}

            num_classes = len(config.ocr.alphabet) + 3
            self.SOS_IDX = num_classes - 3
            self.EOS_IDX = num_classes - 2
            self.PAD_IDX = num_classes - 1

            self.model = PlateRecognizer(
                num_classes=num_classes,
                hidden_dim=config.model.hidden_size,
                max_len=config.model.max_len,
                nhead=config.model.nhead,
                num_layers=config.model.num_layers,
            )
        else:
            self.total_steps = total_steps
            self.tf_ratio = 1.0

            self.alphabet = alphabet
            self.char2idx = {c: i for i, c in enumerate(alphabet)}
            self.idx2char = {i: c for c, i in self.char2idx.items()}
            self.SOS_IDX = len(alphabet)
            self.EOS_IDX = len(alphabet) + 1
            self.PAD_IDX = len(alphabet) + 2

            self.model = PlateRecognizer(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                max_len=max_len,
                nhead=nhead,
                num_layers=num_layers,
            )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.PAD_IDX, label_smoothing=0.1
        )

    def forward(self, x: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, tgt)

    def training_step(self, batch, batch_idx):
        images, texts = batch
        device = images.device
        tgt_input, tgt_output = self.prepare_target(texts, device)

        if self.current_epoch > 2:
            self.tf_ratio = max(0.5, 1.0 - 0.05 * (self.current_epoch - 2))

        logits = self.model(images, tgt_input, tf_ratio=self.tf_ratio)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), tgt_output.view(-1))

        if batch_idx % 100 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                decoded = self.tokens_to_string(preds)
                fit_acc = sum(1 for t, d in zip(texts, decoded) if t == d) / len(texts)
            self.log("train_fit_acc", fit_acc, prog_bar=True, batch_size=len(texts))

        self.log(
            "train_loss", loss, prog_bar=True, batch_size=len(texts), sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch
        preds = self.model(images)

        decoded = self.tokens_to_string(preds)

        cer = jiwer.cer(texts, decoded)
        self.log("val_cer", cer, prog_bar=True, sync_dist=True, batch_size=len(texts))
        return {"val_cer": cer}

    def prepare_target(self, texts: list[str], device: torch.device):
        tgt_input_list = []
        tgt_output_list = []

        for text in texts:
            chars = [self.char2idx[c] for c in text if c in self.char2idx]
            tgt_input_list.append([self.SOS_IDX] + chars)
            tgt_output_list.append(chars + [self.EOS_IDX])

        max_len = min(max(len(t) for t in tgt_input_list), self.model.max_len)
        batch_size = len(texts)

        tgt_input = torch.full(
            (batch_size, max_len), self.PAD_IDX, dtype=torch.long, device=device
        )
        tgt_output = torch.full(
            (batch_size, max_len), self.PAD_IDX, dtype=torch.long, device=device
        )

        for i, (inp, out) in enumerate(zip(tgt_input_list, tgt_output_list)):
            length = min(len(inp), max_len)
            tgt_input[i, :length] = torch.tensor(inp[:length], device=device)
            tgt_output[i, :length] = torch.tensor(out[:length], device=device)

        return tgt_input, tgt_output

    def tokens_to_string(self, tokens: torch.Tensor) -> list[str]:
        decoded = []
        for seq in tokens:
            chars = []
            for idx in seq:
                idx = idx.item() if torch.is_tensor(idx) else int(idx)
                if idx == self.EOS_IDX or idx == self.PAD_IDX:
                    break
                if idx in self.idx2char:
                    chars.append(self.idx2char[idx])
            decoded.append("".join(chars))
        return decoded

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-5, weight_decay=0.05, betas=(0.9, 0.98), eps=1e-6
        )

        if hasattr(self, "hparams") and hasattr(self.hparams, "config"):
            steps_per_epoch = self.total_steps // self.hparams.config.trainer.max_epochs
            warmup_steps = 3 * steps_per_epoch

            warmup = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=self.total_steps - warmup_steps, eta_min=1e-6
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}
