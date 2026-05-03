import os
import platform

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

import sys

sys.path.append("src")

from train_ocr_v2 import ocr_collate_fn

from data.ocr_dataset_v2 import OCRDatasetV2
from pl_modules.ocr_module_v3 import OCRModuleV3


if platform.system() == "Darwin":
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="../configs", config_name="ocr_v3", version_base="1.3")
def main(cfg: DictConfig):
    train_ocr(cfg=cfg)


def train_ocr(cfg: DictConfig):
    train_ds = OCRDatasetV2(
        root_dir=f"{cfg.data.path}/train/img",
        alphabet=cfg.ocr.alphabet,
        img_h=cfg.data.img_h,
        img_w=cfg.data.img_w,
        is_train=True,
    )
    val_ds = OCRDatasetV2(
        root_dir=f"{cfg.data.path}/val/img",
        alphabet=cfg.ocr.alphabet,
        img_h=cfg.data.img_h,
        img_w=cfg.data.img_w,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        collate_fn=ocr_collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        collate_fn=ocr_collate_fn,
        pin_memory=True,
    )

    total_steps = len(train_loader) * cfg.trainer.max_epochs
    model = OCRModuleV3(cfg, total_steps=total_steps)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="mlartifacts/ocr_v3/",
        filename="crnn-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_cer", patience=cfg.trainer.patience, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        logger=mlf_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=5.0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
