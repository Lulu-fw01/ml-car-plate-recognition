import mlflow
from models.yolo_wrap import YOLOWrapper, create_yolo_data_yaml


def train_and_log_detection(cfg):
    data_root = cfg.data.download.output_dir
    data_yaml = "data.yaml"
    create_yolo_data_yaml(data_root, data_yaml)

    model = YOLOWrapper(
        model_name=cfg.model.yolo.model_name,
    )

    best_model_path = model.train(
        data_yaml=data_yaml,
        epochs=cfg.trainer.max_epochs,
        imgsz=cfg.model.yolo.imgsz,
        batch=cfg.trainer.batch_size,
    )

    map50 = model.val(data_yaml=data_yaml, imgsz=cfg.model.yolo.imgsz)

    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    with mlflow.start_run(run_name="yolo-baseline"):
        mlflow.log_params(
            {
                "model": cfg.model.yolo.model_name,
                "from_scratch": cfg.model.yolo.from_scratch,
                "imgsz": cfg.model.yolo.imgsz,
                "epochs": cfg.trainer.max_epochs,
                "batch_size": cfg.trainer.batch_size,
            }
        )
        mlflow.log_metric("val_map50", map50)
        mlflow.log_artifact(str(best_model_path), artifact_path="models")

    print(f" Baseline training complete. mAP@0.5 = {map50:.4f}")
    return best_model_path
