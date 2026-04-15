import os
import platform

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import sys

sys.path.append("src")

from data.ocr_dataset_v2 import OCRDatasetV2
from pl_modules.ocr_module_v2 import OCRModuleV2


if platform.system() == "Darwin":
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="../configs", config_name="ocr_v2", version_base="1.3")
def main(cfg: DictConfig):
    train_ds = OCRDatasetV2(
        root_dir=f"{cfg.data.path}/train/img",
        alphabet=cfg.ocr.alphabet,
        img_h=cfg.data.img_h,
        img_w=cfg.data.img_w,
    )
    val_ds = OCRDatasetV2(
        root_dir=f"{cfg.data.path}/val/img",
        alphabet=cfg.ocr.alphabet,
        img_h=cfg.data.img_h,
        img_w=cfg.data.img_w,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
    )

    model = OCRModuleV2(cfg)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="mlartifacts/ocr_v2/",
        filename="crnn-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    # trainer = Trainer(
    #     logger=mlf_logger,
    #     callbacks=[checkpoint_callback],
    #     max_epochs=cfg.trainer.max_epochs,
    #     accelerator=device,
    #     devices=1,
    # )
    trainer = Trainer(
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=5.0,
        precision="16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
