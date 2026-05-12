import hydra
import mlflow
from omegaconf import DictConfig
from pathlib import Path
from ultralytics import YOLO

import random
import numpy as np
import torch


def seed_everything_yolo(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_yolo_data_yaml(data_root: str, output_path: str):
    data_root = Path(data_root).resolve().as_posix()
    content = f"""path: {data_root}
train: train/images
val: val/images
nc: 1
names: ['license_plate']
"""
    Path(output_path).write_text(content)


@hydra.main(config_path="../configs", config_name="detect_v2", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything_yolo(42)
    data_yaml = "data_detect_v2.yaml"
    create_yolo_data_yaml(cfg.data.path, data_yaml)

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    with mlflow.start_run(run_name="yolov10s-fast"):
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "imgsz": cfg.model.imgsz,
                "epochs": cfg.trainer.max_epochs,
                "batch_size": cfg.trainer.batch_size,
                "patience": cfg.trainer.patience,
                "amp": cfg.trainer.amp,
            }
        )

        model = YOLO(f"{cfg.model.name}.pt")

        _ = model.train(
            data=data_yaml,
            epochs=cfg.trainer.max_epochs,
            imgsz=cfg.model.imgsz,
            batch=cfg.trainer.batch_size,
            device=cfg.trainer.get("device", "auto"),
            project="runs/detect_v2",
            name="yolov10s",
            exist_ok=True,
            verbose=True,
            amp=cfg.trainer.amp,
            patience=cfg.trainer.patience,
            workers=0,
            cache=True,
            close_mosaic=10,
            **cfg.augmentations,
        )

        val_results = model.val(data=data_yaml, imgsz=cfg.model.imgsz)
        map50 = val_results.box.map50

        mlflow.log_metric("val_map50", map50)

        best_model_path = Path("runs/detect_v2/yolov10s/weights/best.pt")
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="models")

        print(f"Training complete. mAP@0.5 = {map50:.4f}")
        return best_model_path


if __name__ == "__main__":
    main()
