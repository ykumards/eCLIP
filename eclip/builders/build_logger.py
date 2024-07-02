import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl


def build_logger(cfg: DictConfig):
    # This path points to the dynamically created directory by Hydra
    run_dir = os.getcwd()
    model_save_path = Path(run_dir) / "saved_model"
    cfg.model.checkpoint_path = model_save_path
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=run_dir,
        project=cfg.wandb_project_name,
        config=OmegaConf.to_container(cfg),  # log the configuration parameters to wandb
    )
    return wandb_logger, cfg
