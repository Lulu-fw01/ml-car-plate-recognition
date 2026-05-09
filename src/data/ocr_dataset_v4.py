from pathlib import Path
import numpy as np
import torch
import albumentations as A
from PIL import Image


class OCRDatasetV4(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        alphabet: str,
        img_h: int = 48,
        img_w: int = 160,
        is_train: bool = True,
    ):
        self.root = Path(root_dir)
        self.img_paths = sorted(self.root.glob("*.png"))
        self.alphabet = alphabet
        self.img_h = img_h
        self.img_w = img_w
        self.is_train = is_train

        self.train_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.4),
                A.RandomResizedCrop(
                    size=(64, 160), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                A.CoarseDropout(max_holes=4, max_height=8, max_width=16, p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("L")
        image = np.array(image)

        if self.is_train:
            augmented = self.train_transform(image=image)
            image = augmented["image"]

        if image.shape[0] != self.img_h or image.shape[1] != self.img_w:
            from PIL import Image as PILImage

            pil_image = PILImage.fromarray(image.astype(np.uint8))
            pil_image = pil_image.resize(
                (self.img_w, self.img_h), PILImage.Resampling.BILINEAR
            )
            image = np.array(pil_image)

        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).unsqueeze(0)

        label = img_path.stem.upper()

        return image, label

    def get_raw_text(self, idx: int) -> str:
        return self.img_paths[idx].stem.upper()
