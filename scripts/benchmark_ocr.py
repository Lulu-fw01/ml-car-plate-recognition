import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.cer import get_cer
from utils.save_errors import save_errors_to_file
from utils.confusion_matrix import update_confusion_matrix, save_to_csv, save_to_png
from utils.prf import calculate_prf
from utils.accuracy import calculate_per_position_accuracy

sys.path.append("src")

from pl_modules.ocr_module_v1 import OCRModuleV1
from pl_modules.ocr_module_v2 import OCRModuleV2
from pl_modules.ocr_module_v3 import OCRModuleV3
from pl_modules.ocr_module_v4 import OCRModuleV4


ALPHABET_V1_V2_V3 = "0123456789ABEKMHOPCTYX_"
ALPHABET_V4 = "0123456789ABEKMHOPCTYX"
LABELS_V1_V2_V3 = list(ALPHABET_V1_V2_V3) + ["<DEL>", "<INS>"]
LABELS_V4 = list(ALPHABET_V4) + ["<SOS>", "<EOS>", "<PAD>"]
BLANK_IDX = len(ALPHABET_V1_V2_V3) - 1

MODEL_V1_PATH = "mlartifacts/ocr/ocr-v1-epoch=04-val_acc=0.9583.ckpt"
MODEL_V2_PATH = "mlartifacts/ocr_v2/crnn-epoch=03-val_cer=0.0178.ckpt"
MODEL_V3_PATH = "mlartifacts/ocr_v3/crnn-epoch=03-val_cer=0.0080.ckpt"
MODEL_V4_PATH = "mlartifacts/ocr_v4/transformer-epoch=22-val_cer=0.0216.ckpt"

TEST_DIR = "data/raw/ocr/test/img"


class OCRBenchmark:
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.img_paths = sorted(self.test_dir.glob("*.png"))

        self.transform_v1 = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.transform_v2 = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (64, 160), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.transform_v3 = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (64, 160), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.transform_v4 = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize(
                    (64, 160), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.device = "cpu"
        self.model_v1 = None
        self.model_v2 = None
        self.model_v3 = None
        self.model_v4 = None

    def load_model_v1(self):
        self.model_v1 = OCRModuleV1.load_from_checkpoint(
            MODEL_V1_PATH, map_location=self.device, weights_only=False
        )
        print(f"OCR v1 model loaded from {MODEL_V1_PATH}")

    def load_model_v2(self):
        self.model_v2 = OCRModuleV2.load_from_checkpoint(
            MODEL_V2_PATH, map_location=self.device, weights_only=False
        )
        print(f"OCR v2 model loaded from {MODEL_V2_PATH}")

    def load_model_v3(self):
        self.model_v3 = OCRModuleV3.load_from_checkpoint(
            MODEL_V3_PATH, map_location=self.device, weights_only=False
        )
        print(f"OCR v3 model loaded from {MODEL_V3_PATH}")

    def load_model_v4(self):
        self.model_v4 = OCRModuleV4(
            alphabet=ALPHABET_V4,
            num_classes=25,
            hidden_dim=256,
            max_len=9,
            nhead=8,
            num_layers=3,
        )
        checkpoint = torch.load(
            MODEL_V4_PATH, map_location=self.device, weights_only=False
        )
        self.model_v4.load_state_dict(checkpoint["state_dict"])
        print(f"OCR v4 model loaded from {MODEL_V4_PATH}")

    def predict_v1(self, tensor: torch.Tensor) -> str:
        self.model_v1.eval()
        device = next(self.model_v1.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = self.model_v1.model(tensor)
            preds = logits.argmax(dim=-1).squeeze()

        chars = []
        for idx in preds:
            idx_item = idx.item()
            if idx_item == BLANK_IDX:
                break
            if idx_item < len(ALPHABET_V1_V2_V3):
                chars.append(ALPHABET_V1_V2_V3[idx_item])

        return "".join(chars)

    def predict_v2(self, tensor: torch.Tensor) -> str:
        self.model_v2.eval()
        device = next(self.model_v2.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = self.model_v2(tensor)
            preds = logits.argmax(dim=-1).squeeze()

        chars = []
        prev = -1

        for idx in preds:
            idx_item = idx.item()
            if idx_item != prev and idx_item != BLANK_IDX:
                if idx_item < len(ALPHABET_V1_V2_V3):
                    chars.append(ALPHABET_V1_V2_V3[idx_item])
            prev = idx_item

        return "".join(chars)

    def predict_v3(self, tensor: torch.Tensor) -> str:
        self.model_v3.eval()
        device = next(self.model_v3.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = self.model_v3(tensor)
            preds = logits.argmax(dim=-1).squeeze()

        chars = []
        prev = -1

        for idx in preds:
            idx_item = idx.item()
            if idx_item != prev and idx_item != BLANK_IDX:
                if idx_item < len(ALPHABET_V1_V2_V3):
                    chars.append(ALPHABET_V1_V2_V3[idx_item])
            prev = idx_item

        return "".join(chars)

    def predict_v4(self, tensor: torch.Tensor) -> str:
        self.model_v4.eval()
        device = next(self.model_v4.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            tokens = self.model_v4(tensor)

        return self.model_v4.tokens_to_string(tokens)[0]

    def benchmark_model(self, model_name: str) -> dict:
        predict_fn = getattr(self, f"predict_{model_name}")
        transform_fn = getattr(self, f"transform_{model_name}")

        labels = LABELS_V1_V2_V3 if model_name in ["v1", "v2", "v3"] else LABELS_V4

        predictions = []
        ground_truths = []
        errors = []
        confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

        total_s = 0
        total_d = 0
        total_i = 0
        total_n = 0

        for i in range(min(10, len(self.img_paths))):
            img = Image.open(self.img_paths[i]).convert("L")
            tensor = transform_fn(img).unsqueeze(0)
            predict_fn(tensor)

        print(f"Benchmarking {model_name} on {len(self.img_paths)} images")
        start_time = time.time()

        for img_path in tqdm(self.img_paths, desc=f"Running {model_name}"):
            img = Image.open(img_path).convert("L")
            tensor = transform_fn(img).unsqueeze(0)

            pred = predict_fn(tensor)
            target = img_path.stem.upper()

            predictions.append(pred)
            ground_truths.append(target)

            cer, s, d, i = get_cer(pred, target)
            total_s += s
            total_d += d
            total_i += i
            total_n += len(target)

            confusion_matrix = update_confusion_matrix(
                confusion_matrix, pred, target, labels
            )

            if pred != target:
                errors.append(
                    {
                        "filename": img_path.name,
                        "ground_truth": target,
                        "prediction": pred,
                        "cer": round(cer, 4),
                        "s": s,
                        "d": d,
                        "i": i,
                    }
                )

        elapsed_time = time.time() - start_time

        full_accuracy = sum(p == t for p, t in zip(predictions, ground_truths)) / len(
            predictions
        )
        overall_cer = (total_s + total_d + total_i) / total_n if total_n > 0 else 0

        per_position = calculate_per_position_accuracy(predictions, ground_truths)
        prf_metrics = calculate_prf(predictions, ground_truths)

        return {
            "model": model_name,
            "num_samples": len(predictions),
            "timing": {
                "total_time_sec": round(elapsed_time, 2),
                "avg_time_ms": round(elapsed_time / len(predictions) * 1000, 2),
                "fps": round(len(predictions) / elapsed_time, 2),
            },
            "metrics": {
                "full_accuracy": round(full_accuracy, 4),
                "cer": round(overall_cer, 4),
                "cer_s": total_s,
                "cer_d": total_d,
                "cer_i": total_i,
                "cer_n": total_n,
                **prf_metrics,
            },
            "per_position_accuracy": per_position,
            "confusion_matrix": confusion_matrix,
            "errors": errors,
            "labels": labels,
        }

    def run_comparison(self) -> dict:
        results = {
            "v1": self.benchmark_model("v1"),
            "v2": self.benchmark_model("v2"),
            "v3": self.benchmark_model("v3"),
            "v4": self.benchmark_model("v4"),
        }
        return results

    def save_results(self, results: dict, output_dir: str = "benchmark_results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        json_results = {}
        for key in results:
            json_results[key] = {
                "num_samples": results[key]["num_samples"],
                "timing": results[key]["timing"],
                "metrics": results[key]["metrics"],
                "per_position_accuracy": results[key]["per_position_accuracy"],
            }

        with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        for key in results:
            save_errors_to_file(
                results[key]["errors"], output_dir / f"{key}_errors.csv"
            )
            save_to_csv(
                results[key]["confusion_matrix"],
                results[key]["labels"],
                output_dir / f"confusion_matrix_{key}.csv",
            )
            save_to_png(
                results[key]["confusion_matrix"],
                results[key]["labels"],
                output_dir / f"confusion_matrix_{key}.png",
            )

        print(f"\nResults saved to {output_dir}/")

    def print_summary(self, results: dict):
        print("\n" + "=" * 100)
        print("OCR Models Comparison")
        print("=" * 100)

        model_keys = list(results.keys())
        header = f"{'Metric':<25}" + "".join([f"{f'OCR {k}':<18}" for k in model_keys])
        print(f"\n{header}")
        print("-" * (25 + 18 * len(model_keys)))

        print(f"\n{'TIMING':-^{25 + 18 * len(model_keys)}}")
        timing_metrics = ["total_time_sec", "avg_time_ms", "fps"]
        for metric in timing_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key]['timing'][metric]:<18.2f}"
            print(row)

        print(f"\n{'MAIN METRICS':-^{25 + 18 * len(model_keys)}}")
        main_metrics = ["full_accuracy", "cer", "precision", "recall", "f1_score"]
        for metric in main_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key]['metrics'][metric]:<18.4f}"
            print(row)

        print(f"\n{'CER BREAKDOWN':-^{25 + 18 * len(model_keys)}}")
        cer_metrics = ["cer_s", "cer_d", "cer_i", "cer_n"]
        for metric in cer_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key]['metrics'][metric]:<18}"
            print(row)

        print(f"\n{'ERRORS':-^{25 + 18 * len(model_keys)}}")
        row = f"{'Total errors':<25}"
        for key in model_keys:
            row += f"{len(results[key]['errors']):<18}"
        print(row)

        row = f"{'Error rate (%)':<25}"
        for key in model_keys:
            rate = len(results[key]["errors"]) / results[key]["num_samples"] * 100
            row += f"{rate:<18.2f}"
        print(row)

        print("\n" + "=" * 100)


def main():
    print("OCR Model Benchmark")
    print("=" * 50)
    print(f"Test directory: {TEST_DIR}")

    benchmark = OCRBenchmark(TEST_DIR)

    benchmark.load_model_v1()
    benchmark.load_model_v2()
    benchmark.load_model_v3()
    benchmark.load_model_v4()

    results = benchmark.run_comparison()

    benchmark.print_summary(results)
    benchmark.save_results(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
