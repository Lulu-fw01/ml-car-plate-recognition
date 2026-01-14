import os
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import snapshot_download
import zipfile

import hydra
from omegaconf import DictConfig, OmegaConf


def _get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent.resolve()


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_to}...")

    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        raise RuntimeError(f"failed to extract {archive_path}: {e}")


def download_dataset(
    repo: str,
    dataset_config: Dict[str, Any],
    output_root: Path,
) -> None:
    name = dataset_config["name"]
    output_subdir = dataset_config["output_subdir"]
    repo_id = repo + "/" + name

    extract_to = output_root / output_subdir

    os.makedirs(output_root, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=extract_to,
            local_dir_use_symlinks=False,
        )
        _extract_archive(extract_to / "test.zip", extract_to)
        _extract_archive(extract_to / "train.zip", extract_to)
        _extract_archive(extract_to / "val.zip", extract_to)

    except Exception as e:
        print(f"\n error downloading {repo_id}: {e}")
    print("OK")


@hydra.main(
    version_base="1.3",
    config_path=str(_get_repo_root() / "configs"),
    config_name="config",
)
def download_all(cfg: DictConfig) -> None:
    output_dir = Path(cfg.data.output_dir)
    repo = cfg.data.repo

    for ds_key, ds_cfg in cfg.data.datasets.items():
        download_dataset(repo, OmegaConf.to_container(ds_cfg), output_dir)


def cli() -> None:
    download_all()


if __name__ == "__main__":
    download_all()
