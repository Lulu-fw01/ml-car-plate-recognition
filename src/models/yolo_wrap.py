from pathlib import Path
from ultralytics import YOLO
from mlflow import active_run
import mlflow


def create_yolo_data_yaml(data_root: str, output_path: str):
    data_root = Path(data_root).resolve().as_posix()
    content = f"""path: {data_root}
train: train/images
val: val/images
nc: 1
names: ['license_plate']
"""
    Path(output_path).write_text(content)


class YOLOWrapper:
    def __init__(self, model_name: str = "yolov5s"):
        self.model = YOLO(f"{model_name}.pt")
        self.mlflow_logger = None

    def train(
        self, data_yaml: str, epochs: int, imgsz: int, batch: int, project: str = "runs"
    ):
        run = active_run()
        if run:
            self.mlflow_logger = run.info.run_id

        self.model.train(
            device="mps",
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name="baseline",
            exist_ok=True,
            verbose=True,
            mosaic=0.0,
            mixup=0.0,
        )
        return Path(project) / "baseline" / "weights" / "best.pt"

    def val(self, data_yaml: str, imgsz: int):
        metrics = self.model.val(data=data_yaml, imgsz=imgsz)
        return metrics.box.map50

    def _log_metrics_to_mlflow(self, trainer):
        metrics = trainer.metrics
        epoch = trainer.epoch + 1

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=epoch)

    def add_mlflow_callback(self):
        if self.mlflow_logger is not None:
            self.model.add_callback("on_fit_epoch_end", self._log_metrics_to_mlflow)
