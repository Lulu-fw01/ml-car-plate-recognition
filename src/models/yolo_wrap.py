from pathlib import Path
from ultralytics import YOLO


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
        self.model = YOLO(f"{model_name}.yaml")

    def train(
        self, data_yaml: str, epochs: int, imgsz: int, batch: int, project: str = "runs"
    ):
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name="baseline",
            exist_ok=True,
            verbose=False,
        )
        return Path(project) / "baseline" / "weights" / "best.pt"

    def val(self, data_yaml: str, imgsz: int):
        metrics = self.model.val(data=data_yaml, imgsz=imgsz)
        return metrics.box.map50
