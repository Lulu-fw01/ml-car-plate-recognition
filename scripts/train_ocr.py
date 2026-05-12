import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from data.ocr_dataset import OCRDataset
from pl_modules.ocr_module_v1 import OCRModuleV1
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="../configs", config_name="ocr", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(42)
    train_ds = OCRDataset(cfg.model.alphabet, cfg.data.path + "/train/img")
    val_ds = OCRDataset(cfg.model.alphabet, cfg.data.path + "/val/img")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.trainer.batch_size, num_workers=cfg.trainer.num_workers
    )

    model = OCRModuleV1(cfg)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
    )

    cp_callback = ModelCheckpoint(
        dirpath="mlartifacts/ocr/",
        filename="ocr-v1-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    trainer = Trainer(
        logger=mlf_logger,
        callbacks=[cp_callback],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
