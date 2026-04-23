import torch
import torch.nn as nn
import math

from torchvision.models import resnet34, ResNet34_Weights


class ResNet34CRNN(nn.Module):
    def __init__(
        self, num_classes: int, hidden_size: int = 256, num_lstm_layers: int = 2
    ):
        super().__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # cсверточный слой
        self.bn1 = resnet.bn1  # норм
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._fix_stride(self.layer3, stride=(2, 1))
        self._fix_stride(self.layer4, stride=(2, 1))

        self.projection = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_lstm_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.register_buffer(
            "pos_encoding", self._make_positional_encoding(max_len=128, d_model=256)
        )

    def _make_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def _fix_stride(self, layer, stride):
        layer[0].conv1.stride = stride
        if layer[0].downsample is not None:
            layer[0].downsample[0].stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, H, W]

        x = torch.mean(x, dim=2)  # [B, 512, W]
        x = x.permute(0, 2, 1)  # [B, W, 512] - формат для LSTM (Batch, Seq, Features)

        x = self.projection(x)

        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.fc(x)  # [B, W, num_classes]

        return x.permute(1, 0, 2)
