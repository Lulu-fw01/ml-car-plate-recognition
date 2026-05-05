import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


class PlateRecognizer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        max_len: int = 9,
        nhead: int = 8,
        num_layers: int = 3,
        vis_seq_len: int = 100,
    ):
        super().__init__()
        resnet = resnet34(weights=None)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # cверточный слой
        self.bn1 = nn.BatchNorm2d(64, dtype=torch.float32)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self._fix_stride(self.layer3, stride=(2, 1))
        self._fix_stride(self.layer4, stride=(2, 1))

        self.num_classes = num_classes
        self.max_len = max_len

        self.proj = nn.Linear(512, hidden_dim)

        # PE для визуальной последовательности (энкодер)
        self.vis_pe = nn.Parameter(torch.randn(1, vis_seq_len, hidden_dim) * 0.02)

        # 2. Transformer Encoder (для глобального контекста)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. Projection & Embeddings
        self.char_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        # Positional encoding для декодера
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        self.SOS_IDX = 22
        self.EOS_IDX = 23
        self.PAD_IDX = 24

        self.register_buffer("blank_mask", torch.tensor(0.0))

    def _fix_stride(self, layer, stride):
        layer[0].conv1.stride = stride
        if layer[0].downsample is not None:
            layer[0].downsample[0].stride = stride

    def forward(self, x: torch.Tensor, tgt: torch.Tensor = None, tf_ratio: float = 1.0):
        B = x.size(0)

        # 1. Полная CNN-часть
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, H', W']

        # 2. Pooling + Projection
        x = F.adaptive_avg_pool2d(x, (1, x.size(3))).squeeze(2)  # [B, 512, W']
        x = x.permute(0, 2, 1)  # [B, W', 512]
        feats = self.proj(x)  # [B, W', hidden_dim]

        # 3. Encoder с PE
        seq_len = feats.size(1)
        if seq_len != self.vis_pe.size(1):
            pe = self.vis_pe.permute(0, 2, 1)  # [1, hidden_dim, max_len]
            pe = F.interpolate(pe, size=seq_len, mode="linear", align_corners=False)
            pe = pe.permute(0, 2, 1)  # [1, seq_len, hidden_dim]
        else:
            pe = self.vis_pe[:, :seq_len, :]
        feats = feats + pe
        memory = self.encoder(feats)  # [B, W', hidden_dim]

        if self.training and tgt is not None and tf_ratio < 1.0:
            B, seq_len = tgt.shape
            mask = torch.rand(B, seq_len, device=tgt.device) < tf_ratio  # [B, seq_len]

            with torch.no_grad():
                greedy_preds = self._greedy_decode(
                    memory, max_len=seq_len
                )  # [B, seq_len]

            tgt = torch.where(mask, tgt, greedy_preds)

        if self.training and tgt is not None:
            # Teacher Forcing
            tgt_emb = self.char_embed(tgt) + self.pos_embed[:, : tgt.size(1), :]
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(x.device)
            tgt_pad_mask = (tgt == self.PAD_IDX).bool()

            out = self.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask
            )
            return self.fc_out(out)  # [B, seq_len, num_classes]
        else:
            # Autoregressive inference
            preds = []
            decoder_input = torch.full(
                (B, 1), self.SOS_IDX, dtype=torch.long, device=x.device
            )

            for _ in range(self.max_len):
                tgt_emb = (
                    self.char_embed(decoder_input)
                    + self.pos_embed[:, : decoder_input.size(1), :]
                )
                out = self.decoder(tgt_emb, memory)
                logits = self.fc_out(out[:, -1, :])
                pred = logits.argmax(dim=-1, keepdim=True)
                preds.append(pred)
                decoder_input = torch.cat([decoder_input, pred], dim=1)

                if (pred == self.EOS_IDX).all():
                    break

            return torch.cat(preds, dim=1)  # [B, pred_len]

    def _greedy_decode(self, memory: torch.Tensor, max_len: int) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device

        # Start with SOS token
        decoder_input = torch.full(
            (B, 1), self.SOS_IDX, dtype=torch.long, device=device
        )
        predictions = []

        for step in range(max_len):
            # Embed + positional encoding
            tgt_emb = (
                self.char_embed(decoder_input)
                + self.pos_embed[:, : decoder_input.size(1), :]
            )

            # Forward through decoder
            out = self.decoder(tgt_emb, memory)

            # Predict next token
            logits = self.fc_out(out[:, -1, :])  # [B, num_classes]
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            predictions.append(next_token)

            # Append to decoder input for next step
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Early stopping if all sequences generated EOS
            if (next_token == self.EOS_IDX).all():
                # Pad remaining steps if needed
                remaining = max_len - len(predictions)
                if remaining > 0:
                    pad_tokens = torch.full(
                        (B, remaining), self.PAD_IDX, dtype=torch.long, device=device
                    )
                    predictions.append(pad_tokens)
                break

        # Concatenate predictions: list of [B, 1] to [B, max_len]
        if len(predictions) < max_len:
            # Pad if we stopped early
            remaining = max_len - len(predictions)
            pad_tokens = torch.full(
                (B, remaining), self.PAD_IDX, dtype=torch.long, device=device
            )
            predictions.append(pad_tokens)

        return torch.cat(predictions, dim=1)  # [B, max_len]

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask
