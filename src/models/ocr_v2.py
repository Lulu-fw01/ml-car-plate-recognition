import torch
import torch.nn as nn
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
        self.maxpool = nn.Identity()

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self._fix_stride(self.layer3, stride=(2, 1))
        self._fix_stride(self.layer4, stride=(2, 1))

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_lstm_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

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

        x, _ = self.lstm(x)
        x = self.fc(x)  # [B, W, num_classes]

        return x.permute(1, 0, 2)
