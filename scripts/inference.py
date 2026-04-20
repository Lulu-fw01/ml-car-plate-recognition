import argparse
import os

import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from pl_modules.ocr_module_v1 import OCRModuleV1

ALPHABET = "0123456789ABEKMHOPCTYX_"
PAD_IDX = len(ALPHABET) - 1


def load_detection_model(path: str):
    model = YOLO(path)
    return model


def load_ocr_model(path: str):
    model = OCRModuleV1.load_from_checkpoint(
        path, map_location="cpu", weights_only=False
    )
    model.eval()
    return model


def detect_plate(model, image_path: str, conf_thres: float = 0.5):
    """return car plate coordinates"""
    results = model(image_path, conf=conf_thres)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return x1, y1, x2, y2


def crop_plate(image_path: str, coords):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = coords
    cropped = image[y1:y2, x1:x2]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped)


def prepare_for_ocr(pil_image, img_h=32, img_w=128):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transform(pil_image).unsqueeze(0)  # (1, 1, 32, 128)


def recognize_text(ocr_model, tensor):
    with torch.no_grad():
        logits = ocr_model.model(tensor)  # (1, 9, 23)

        activations = ocr_model.model.cnn[:-1](tensor)  # (1, 128, H/4, W/4)
        print(f"Shape: {activations.shape}")  # Должно быть (1, 128, ~6, ~24)

        # Усредняем по каналам (dim=1), чтобы получить 2D карту
        # (1, 128, H, W) -> (1, H, W)
        feature_map = activations[0].mean(dim=0)  # (H/4, W/4)
        plt.figure(figsize=(10, 4))
        plt.imshow(feature_map.detach().cpu(), cmap="hot", aspect="auto")
        plt.title("Feature Map (before AdaptiveAvgPool)")
        plt.colorbar(label="Activation")
        plt.show()

        preds = logits.argmax(dim=-1).squeeze()  # (9,)
    text = "".join(ALPHABET[i] for i in preds if i != PAD_IDX)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--detect-path", type=str)
    parser.add_argument("--ocr-path", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.image_path), f"Image not found: {args.image_path}"
    assert os.path.exists(args.detect_path), (
        f"Detection weights not found: {args.detect_path}"
    )
    assert os.path.exists(args.ocr_path), f"OCR checkpoint not found: {args.ocr_path}"

    print(" Loading detection model")
    det_model = load_detection_model(args.detect_path)

    print(" Loading OCR model")
    ocr_model = load_ocr_model(args.ocr_path)

    print(" Detecting")
    coords = detect_plate(det_model, args.image_path)
    if coords is None:
        print(" No license plate detected.")
        return

    cropped = crop_plate(args.image_path, coords)
    tensor = prepare_for_ocr(cropped)
    print(" Recognizing")
    text = recognize_text(ocr_model, tensor)

    print(f"Result: {text}")


if __name__ == "__main__":
    main()
