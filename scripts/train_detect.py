import hydra
from pytorch_lightning import seed_everything
from omegaconf import DictConfig
from pl_modules.detect_module import train_and_log_detection


@hydra.main(config_path="../configs", config_name="detect", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(42)
    train_and_log_detection(cfg)


if __name__ == "__main__":
    main()
