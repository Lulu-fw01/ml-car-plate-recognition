from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


class OCRDatasetV2(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, alphabet: str, img_h: int = 32, img_w: int = 128):
        self.root = Path(root_dir)
        self.img_paths = sorted(self.root.glob("*.png"))
        self.alphabet = alphabet

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((img_h, img_w)),  # фиксация размера
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        # data
        # transforms.Compose([
        #     transforms.Grayscale(),
        #     transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        #     transforms.ColorJitter(brightness=0.3, contrast=0.3),
        #     transforms.Resize((32, 160)), # Увеличил ширину
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5])
        # ])

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
