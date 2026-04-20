from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

PAD_TOKEN = "_"
MAX_LENGTH = 9


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, alphabet: str, root_dir: str, img_h: int = 32, img_w: int = 128):
        self.root = Path(root_dir)
        self.img_paths = list(self.root.glob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
        self.pad_idx = len(alphabet) - 1

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = img_path.stem.upper()
        label = label.ljust(MAX_LENGTH, PAD_TOKEN)
        target = torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)
        return image, target
