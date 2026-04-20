import torch.nn as nn


class OCRv1(nn.Module):
    def __init__(self, num_chars=23, max_length=9):
        super().__init__()
        self.max_length = max_length
        self.num_chars = num_chars
        self.pad_token = num_chars - 1

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 32, 3, padding=1
            ),  # 1- цвет не важен, проход окном, 32 признак на выходе, 3x3, отступ 1, чтобы не обрезать изображение.
            nn.ReLU(),  # убираем шум
            nn.MaxPool2d(
                2
            ),  # уменьшаем изображение, делаем его в 2 раза меньше. проходим окном 2x2, выбираем сильный пиксель.
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(
                (1, 32)
            ),  # (B, 128, 1, 32) получаем тензор фиксированного размера. схлопываем пиксели, усредняя
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            128 * 32, max_length * num_chars
        )  # обрабатываем вектор получаем новый вектор. сопоставляем найденные признаки с вероятностями элементов алфавита

    def forward(self, x):
        x = self.cnn(x)  # (B, 128, 1, 32)
        x = self.flatten(x)  # (B, 4096) одномерный вектор
        x = self.fc(x)  # (B, 198)
        x = x.view(
            -1, self.max_length, self.num_chars
        )  # (B, 9, 22) 9 позиций, алфавит из 22 символов. берутся подвекторы размера 22. получаем вероятность каждого символа алфавита на каждую позицию
        return x
