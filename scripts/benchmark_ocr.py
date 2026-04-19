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


ALPHABET = "0123456789ABEKMHOPCTYX_"
LABELS = list(ALPHABET) + ["<DEL>", "<INS>"]
PAD_IDX = len(ALPHABET) - 1

MODEL_V1_PATH = "mlartifacts/ocr/ocr-v1-epoch=06-val_acc=0.9588.ckpt"
MODEL_V2_PATH = "mlartifacts/ocr_v2/crnn-epoch=10-val_cer=0.0202.ckpt"
TEST_DIR = "data/raw/ocr/test/img"
IMG_H = 32
IMG_W = 128


class OCRBenchmark:
    def __init__(self, model_v1_path: str, model_v2_path: str, test_dir: str):
        self.model_v1_path = model_v1_path
        self.model_v2_path = model_v2_path
        self.test_dir = Path(test_dir)
        self.img_paths = sorted(self.test_dir.glob("*.png"))

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.device = "cpu"
        self.model_v1 = None
        self.model_v2 = None

    def load_model_v1(self):
        self.model_v1 = OCRModuleV1.load_from_checkpoint(
            self.model_v1_path, map_location=self.device, weights_only=False
        )
        self.model_v1.eval()
        print(f"OCR v1 model loaded from {self.model_v1_path}")

    def load_model_v2(self):
        self.model_v2 = OCRModuleV2.load_from_checkpoint(
            self.model_v2_path, map_location=self.device, weights_only=False
        )
        self.model_v2.eval()
        print(f"OCR v2 model loaded {self.model_v2_path}")

    def predict_v1(self, tensor: torch.Tensor) -> str:
        with torch.no_grad():
            logits = self.model_v1.model(tensor)
            preds = logits.argmax(dim=-1).squeeze()

        chars = []
        for idx in preds:
            idx_item = idx.item()
            if idx_item == PAD_IDX:
                break
            if idx_item < len(ALPHABET):
                chars.append(ALPHABET[idx_item])

        return "".join(chars)

    def predict_v2(self, tensor: torch.Tensor) -> str:
        with torch.no_grad():
            logits = self.model_v2(tensor)
            preds = logits.argmax(dim=-1)

        if preds.ndim > 1:
            preds = preds[:, 0]

        chars = []
        prev = -1

        for idx in preds:
            idx_item = idx.item()
            if idx_item != prev and idx_item != PAD_IDX:
                if idx_item < len(ALPHABET):
                    chars.append(ALPHABET[idx_item])
            prev = idx_item

        return "".join(chars)

    def benchmark_model(self, model_name: str) -> dict:
        predict_fn = self.predict_v1 if model_name == "v1" else self.predict_v2

        predictions = []
        ground_truths = []
        errors = []
        confusion_matrix = np.zeros((len(ALPHABET) + 2, len(ALPHABET) + 2), dtype=int)

        total_s = 0
        total_d = 0
        total_i = 0
        total_n = 0

        for i in range(min(10, len(self.img_paths))):
            img = Image.open(self.img_paths[i]).convert("L")
            tensor = self.transform(img).unsqueeze(0)
            predict_fn(tensor)

        print(f"Benchmarking {model_name} on {len(self.img_paths)} images")
        start_time = time.time()

        for img_path in tqdm(self.img_paths, desc=f"Running {model_name}"):
            img = Image.open(img_path).convert("L")
            tensor = self.transform(img).unsqueeze(0)

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
                confusion_matrix, pred, target, LABELS
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
        }

    def run_comparison(self) -> dict:
        results = {"v1": self.benchmark_model("v1"), "v2": self.benchmark_model("v2")}
        return results

    def save_results(self, results: dict, output_dir: str = "benchmark_results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        json_results = {
            "v1": {
                "num_samples": results["v1"]["num_samples"],
                "timing": results["v1"]["timing"],
                "metrics": results["v1"]["metrics"],
                "per_position_accuracy": results["v1"]["per_position_accuracy"],
            },
            "v2": {
                "num_samples": results["v2"]["num_samples"],
                "timing": results["v2"]["timing"],
                "metrics": results["v2"]["metrics"],
                "per_position_accuracy": results["v2"]["per_position_accuracy"],
            },
        }

        with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        save_errors_to_file(results["v1"]["errors"], output_dir / "v1_errors.csv")
        save_errors_to_file(results["v2"]["errors"], output_dir / "v2_errors.csv")

        save_to_csv(
            results["v1"]["confusion_matrix"],
            LABELS,
            output_dir / "confusion_matrix_v1.csv",
        )
        save_to_png(
            results["v1"]["confusion_matrix"],
            LABELS,
            output_dir / "confusion_matrix_v1.png",
        )
        save_to_csv(
            results["v2"]["confusion_matrix"],
            LABELS,
            output_dir / "confusion_matrix_v2.csv",
        )
        save_to_png(
            results["v2"]["confusion_matrix"],
            LABELS,
            output_dir / "confusion_matrix_v2.png",
        )

        print(f"\nResults saved to {output_dir}/")

    def print_summary(self, results: dict):
        print("\n" + "=" * 80)
        print("OCR models comparing:")
        print("=" * 80)

        v1 = results["v1"]
        v2 = results["v2"]

        print(f"\n{'Metric':<25} {'OCR v1':<20} {'OCR v2':<20}")
        print("-" * 65)

        print(f"\n{'TIMING':-^65}")
        print(
            f"{'Total time (s)':<25} {v1['timing']['total_time_sec']:<20.2f} {v2['timing']['total_time_sec']:<20.2f}"
        )
        print(
            f"{'Avg time (ms)':<25} {v1['timing']['avg_time_ms']:<20.2f} {v2['timing']['avg_time_ms']:<20.2f}"
        )
        print(f"{'FPS':<25} {v1['timing']['fps']:<20.2f} {v2['timing']['fps']:<20.2f}")

        print(f"\n{'MAIN METRICS':-^65}")
        print(
            f"{'Full Accuracy':<25} {v1['metrics']['full_accuracy']:<20.4f} {v2['metrics']['full_accuracy']:<20.4f}"
        )
        print(
            f"{'CER':<25} {v1['metrics']['cer']:<20.4f} {v2['metrics']['cer']:<20.4f}"
        )
        print(
            f"{'Precision':<25} {v1['metrics']['precision']:<20.4f} {v2['metrics']['precision']:<20.4f}"
        )
        print(
            f"{'Recall':<25} {v1['metrics']['recall']:<20.4f} {v2['metrics']['recall']:<20.4f}"
        )
        print(
            f"{'F1-Score':<25} {v1['metrics']['f1_score']:<20.4f} {v2['metrics']['f1_score']:<20.4f}"
        )

        print(f"\n{'CER BREAKDOWN':-^65}")
        print(
            f"{'Substitutions':<25} {v1['metrics']['cer_s']:<20} {v2['metrics']['cer_s']:<20}"
        )
        print(
            f"{'Deletions':<25} {v1['metrics']['cer_d']:<20} {v2['metrics']['cer_d']:<20}"
        )
        print(
            f"{'Insertions':<25} {v1['metrics']['cer_i']:<20} {v2['metrics']['cer_i']:<20}"
        )
        print(
            f"{'Total chars':<25} {v1['metrics']['cer_n']:<20} {v2['metrics']['cer_n']:<20}"
        )

        print(f"\n{'PER-POSITION ACCURACY':-^65}")
        print(f"{'Position':<10} {'v1 Acc':<15} {'v2 Acc':<15}")
        for pos_v1, pos_v2 in zip(
            v1["per_position_accuracy"], v2["per_position_accuracy"]
        ):
            print(
                f"{pos_v1['position']:<10} {pos_v1['accuracy']:<15.4f} {pos_v2['accuracy']:<15.4f}"
            )

        print(f"\n{'ERRORS':-^65}")
        print(f"{'Total errors':<25} {len(v1['errors']):<20} {len(v2['errors']):<20}")
        print(
            f"{'Error rate (%)':<25} {len(v1['errors']) / v1['num_samples'] * 100:<20.2f} {len(v2['errors']) / v2['num_samples'] * 100:<20.2f}"
        )

        print("\n" + "=" * 80)


def main():
    print("OCR Model Benchmark")
    print("=" * 50)
    print(f"Test directory: {TEST_DIR}")

    benchmark = OCRBenchmark(MODEL_V1_PATH, MODEL_V2_PATH, TEST_DIR)

    benchmark.load_model_v1()
    benchmark.load_model_v2()

    results = benchmark.run_comparison()

    benchmark.print_summary(results)
    benchmark.save_results(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
