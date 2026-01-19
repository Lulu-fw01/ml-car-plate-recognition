import hydra
from omegaconf import DictConfig
from pathlib import Path
from pl_modules.detect_module import train_and_log_detection


def _get_repo_root() -> Path:
    return Path(__file__).parent.parent.resolve()


@hydra.main(config_path="../configs", config_name="detect", version_base="1.3")
def main(cfg: DictConfig):
    train_and_log_detection(cfg)


if __name__ == "__main__":
    main()
