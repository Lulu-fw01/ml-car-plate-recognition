import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, img_h: int, nc: int, nclass: int, nh: int, n_rnn: int = 2):
        super().__init__()
        assert img_h % 16 == 0, "img_h must be divided by 16"

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(nc, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1)),
            # Block 5
            nn.Conv2d(512, 512, 2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, nh, n_rnn, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.cnn(x)  # (B, 512, 1, W')
        x = x.squeeze(2)  # (B, 512, W')
        x = x.permute(2, 0, 1)  # (W', B, 512) = (T, B, C)
        x, _ = self.rnn(x)  # (T, B, 2*nh)
        x = self.fc(x)  # (T, B, nclass)
        return x
