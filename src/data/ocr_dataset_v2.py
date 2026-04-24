from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# class LetterboxResize:
#     def __init__(self, size, fill=255):
#         self.size = size
#         self.fill = fill

#     def __call__(self, img):
#         w, h = img.size
#         target_w, target_h = self.size
#         scale = min(target_w / w, target_h / h)
#         new_w, new_h = int(w * scale), int(h * scale)
#         img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

#         new_img = Image.new('L', (target_w, target_h), self.fill)
#         paste_x = (target_w - new_w) // 2
#         paste_y = (target_h - new_h) // 2
#         new_img.paste(img, (paste_x, paste_y))
#         return new_img


class OCRDatasetV2(torch.utils.data.Dataset):
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

        self.is_train = is_train

        val_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                # transforms.Pad((10, 0, 10, 0), fill=255),
                transforms.Resize(
                    (img_h, img_w), interpolation=transforms.InterpolationMode.BILINEAR
                ),  # фиксация размера
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        train_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                # transforms.Pad((10, 0, 10, 0), fill=255),
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.1, 0.0),
                    scale=(0.92, 1.08),
                    shear=1.5,
                    fill=255,
                ),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.Resize(
                    (img_h, img_w), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.transform = train_transform if is_train else val_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        label = img_path.stem.upper()

        return image, label

    def get_raw_text(self, idx: int) -> str:
        return self.img_paths[idx].stem.upper()
