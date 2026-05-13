import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append("src")

MODEL_V1_PATH = "runs/baseline/weights/best.pt"
MODEL_V2_PATH = "runs/detect/runs/detect_v2/yolov10s/weights/best.pt"
TEST_DIR = "data/raw/detect/test"
IMG_SIZE_V1 = 640
IMG_SIZE_V2 = 320
CONF_V1 = 0.1
CONF_V2 = 0.25
IOU_THRESHOLD = 0.25


class DetectionBenchmark:
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.img_dir = self.test_dir / "images"
        self.label_dir = self.test_dir / "labels"
        self.img_paths = sorted(self.img_dir.glob("*.*"))
        self.label_paths = sorted(self.label_dir.glob("*.txt"))

        self.model_v1 = None
        self.model_v2 = None

    def load_model_v1(self):
        self.model_v1 = YOLO(MODEL_V1_PATH)
        print(f"Detection v1 (YOLOv5s) loaded from {MODEL_V1_PATH}")

    def load_model_v2(self):
        self.model_v2 = YOLO(MODEL_V2_PATH)
        print(f"Detection v2 (YOLOv10s) loaded from {MODEL_V2_PATH}")

    def parse_yolo_label(self, label_path: Path, img_width: int, img_height: int):
        if not label_path.exists():
            return []

        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, cx, cy, w, h = map(float, parts[:5])
                    x1 = (cx - w / 2) * img_width
                    y1 = (cy - h / 2) * img_height
                    x2 = (cx + w / 2) * img_width
                    y2 = (cy + h / 2) * img_height
                    boxes.append({"class": int(cls), "bbox": [x1, y1, x2, y2]})
        return boxes

    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def match_boxes(self, preds, gts, iou_threshold=0.5):
        matched_gt = set()
        matches = []

        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = self.compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                matches.append(
                    {
                        "pred": pred,
                        "gt": gts[best_gt_idx],
                        "iou": best_iou,
                        "correct": True,
                    }
                )
                matched_gt.add(best_gt_idx)
            else:
                matches.append(
                    {
                        "pred": pred,
                        "gt": None,
                        "iou": best_iou if best_gt_idx >= 0 else 0.0,
                        "correct": False,
                    }
                )

        unmatched_gt = [gt for idx, gt in enumerate(gts) if idx not in matched_gt]

        return matches, unmatched_gt

    def predict_boxes(self, img_path: Path, model, img_size: int, conf: float):
        img = cv2.imread(str(img_path))
        if img is None:
            return []

        results = model(img, conf=conf, verbose=False, imgsz=img_size)
        boxes = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                boxes.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class": cls,
                    }
                )

        return boxes

    def benchmark_model(
        self, model_name: str, model, img_size: int, conf: float
    ) -> dict:
        predictions = []
        ground_truths = []
        errors = []
        ious = []

        total_tp = 0
        total_fp = 0
        total_fn = 0

        print(f"\nBenchmarking {model_name} on {len(self.img_paths)} images")

        warmup = min(10, len(self.img_paths))
        for i in range(warmup):
            img_path = self.img_paths[i]
            self.predict_boxes(img_path, model, img_size, conf)

        start_time = time.time()

        for img_path in tqdm(self.img_paths, desc=f"Running {model_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            label_path = self.label_dir / f"{img_path.stem}.txt"
            gt_boxes = self.parse_yolo_label(label_path, img_width, img_height)

            pred_boxes = self.predict_boxes(img_path, model, img_size, conf)

            matches, unmatched_gt = self.match_boxes(
                pred_boxes, gt_boxes, IOU_THRESHOLD
            )

            tp = sum(1 for m in matches if m["correct"])
            fp = sum(1 for m in matches if not m["correct"])
            fn = len(unmatched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            for match in matches:
                ious.append(match["iou"])
                if not match["correct"]:
                    errors.append(
                        {
                            "filename": img_path.name,
                            "type": "fp",
                            "iou": round(match["iou"], 4),
                            "confidence": round(match["pred"]["confidence"], 4),
                            "bbox": match["pred"]["bbox"],
                        }
                    )

            for gt in unmatched_gt:
                errors.append(
                    {"filename": img_path.name, "type": "fn", "bbox": gt["bbox"]}
                )

            predictions.append(
                {
                    "filename": img_path.name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "ious": [m["iou"] for m in matches],
                }
            )
            ground_truths.append(
                {"filename": img_path.name, "num_boxes": len(gt_boxes)}
            )

        elapsed_time = time.time() - start_time

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        valid_ious = [i for i in ious if i > 0]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0

        return {
            "model": model_name,
            "num_samples": len(self.img_paths),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "timing": {
                "total_time_sec": round(elapsed_time, 2),
                "avg_time_ms": round(elapsed_time / len(self.img_paths) * 1000, 2),
                "fps": round(len(self.img_paths) / elapsed_time, 2)
                if elapsed_time > 0
                else 0,
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "mean_iou": round(mean_iou, 4),
                "total_detections": total_tp + total_fp,
                "total_gt_boxes": total_tp + total_fn,
            },
            "iou_distribution": {
                "min": round(min(ious), 4) if ious else 0,
                "max": round(max(ious), 4) if ious else 0,
                "mean": round(np.mean(ious), 4) if ious else 0,
                "median": round(np.median(ious), 4) if ious else 0,
                "std": round(np.std(ious), 4) if ious else 0,
            },
            "errors": errors,
            "all_ious": ious,
        }

    def run_comparison(self) -> dict:
        results = {
            "v1": self.benchmark_model("v1", self.model_v1, IMG_SIZE_V1, CONF_V1),
            "v2": self.benchmark_model("v2", self.model_v2, IMG_SIZE_V2, CONF_V2),
        }
        return results

    def save_visualizations(self, results: dict, output_dir: Path):
        for model_name, result in results.items():
            errors_dir = output_dir / f"detect_{model_name}_errors"
            errors_dir.mkdir(exist_ok=True, parents=True)

            for error in result["errors"]:
                if error["type"] == "fn":
                    continue

                img_path = self.img_dir / error["filename"]
                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                bbox = error["bbox"]
                color = (0, 0, 255) if error["type"] == "fp" else (0, 255, 0)
                cv2.rectangle(
                    img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    2,
                )

                label = f"{error['type'].upper()} IoU:{error['iou']:.2f}"
                if "confidence" in error:
                    label += f" Conf:{error['confidence']:.2f}"

                cv2.putText(
                    img,
                    label,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                cv2.imwrite(str(errors_dir / error["filename"]), img)

            self.save_iou_histogram(result["all_ious"], model_name, output_dir)

            print(
                f"Saved {len([e for e in result['errors'] if e['type'] == 'fp'])} FP images to {errors_dir}"
            )

    def save_iou_histogram(self, ious: list, model_name: str, output_dir: Path):
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=50, edgecolor="black", alpha=0.7)
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.title(f"IoU Distribution - Detection {model_name.upper()}")
        plt.axvline(
            x=IOU_THRESHOLD,
            color="r",
            linestyle="--",
            label=f"Threshold={IOU_THRESHOLD}",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = output_dir / f"detect_{model_name}_iou_hist.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved IoU histogram to {output_path}")

    def save_results(self, results: dict, output_dir: str = "benchmark_results"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        json_results = {}
        for key in results:
            json_results[key] = {
                "model": results[key]["model"],
                "num_samples": results[key]["num_samples"],
                "total_tp": results[key]["total_tp"],
                "total_fp": results[key]["total_fp"],
                "total_fn": results[key]["total_fn"],
                "timing": results[key]["timing"],
                "metrics": results[key]["metrics"],
                "iou_distribution": results[key]["iou_distribution"],
            }

            errors_df = []
            for error in results[key]["errors"]:
                errors_df.append(
                    {
                        "filename": error["filename"],
                        "type": error["type"],
                        "iou": error.get("iou", ""),
                        "confidence": error.get("confidence", ""),
                    }
                )

            errors_path = output_dir / f"detect_{key}_errors.csv"
            with open(errors_path, "w") as f:
                f.write("filename,type,iou,confidence\n")
                for error in errors_df:
                    f.write(
                        f"{error['filename']},{error['type']},{error['iou']},{error['confidence']}\n"
                    )

        with open(output_dir / "detect_benchmark.json", "w") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        self.save_visualizations(results, output_dir)

        print(f"\nResults saved to {output_dir}/")

    def print_summary(self, results: dict):
        print("\n" + "=" * 80)
        print("Detection Models Comparison")
        print("=" * 80)

        model_keys = list(results.keys())
        header = f"{'Metric':<25}" + "".join(
            [f"{f'Detect {k}':<18}" for k in model_keys]
        )
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
        main_metrics = ["precision", "recall", "f1_score", "mean_iou"]
        for metric in main_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key]['metrics'][metric]:<18.4f}"
            print(row)

        print(f"\n{'DETECTION COUNTS':-^{25 + 18 * len(model_keys)}}")
        count_metrics = ["total_tp", "total_fp", "total_fn"]
        for metric in count_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key][metric]:<18}"
            print(row)

        print(f"\n{'IoU DISTRIBUTION':-^{25 + 18 * len(model_keys)}}")
        iou_metrics = ["min", "max", "mean", "median", "std"]
        for metric in iou_metrics:
            row = f"{metric:<25}"
            for key in model_keys:
                row += f"{results[key]['iou_distribution'][metric]:<18.4f}"
            print(row)

        print(f"\n{'ERRORS':-^{25 + 18 * len(model_keys)}}")
        row = f"{'Total errors':<25}"
        for key in model_keys:
            row += f"{len(results[key]['errors']):<18}"
        print(row)

        row = f"{'Error rate (%)':<25}"
        for key in model_keys:
            total_gt = results[key]["total_tp"] + results[key]["total_fn"]
            rate = len(results[key]["errors"]) / total_gt * 100 if total_gt > 0 else 0
            row += f"{rate:<18.2f}"
        print(row)

        print("\n" + "=" * 80)


def main():
    print("Detection Model Benchmark")
    print("=" * 50)
    print(f"Test directory: {TEST_DIR}")
    print(f"Model v1: {MODEL_V1_PATH}")
    print(f"Model v2: {MODEL_V2_PATH}")

    benchmark = DetectionBenchmark(TEST_DIR)

    benchmark.load_model_v1()
    benchmark.load_model_v2()

    results = benchmark.run_comparison()

    benchmark.print_summary(results)
    benchmark.save_results(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
