import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import torch
import wandb
from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment

from eclip.builders import build_logger, build_callbacks, build_datamodule, build_module
from eclip.eval.eval_classification_pneumonia import eval_rsna
from eclip.eval.eval_classification_chexpert import eval_chexpert
from eclip.eval.eval_classification_cxr14x100 import eval_cxr14

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.2", config_path="configs", config_name="default")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)

    wandb_logger, cfg = build_logger(cfg)

    clip_datamodule = build_datamodule(cfg)

    lightning_module = build_module(cfg)

    callbacks = build_callbacks(cfg)

    strat = "auto" if cfg.num_gpus < 2 else "ddp_find_unused_parameters_true"
    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        limit_val_batches=cfg.limit_val_batches,
        limit_train_batches=cfg.limit_train_batches,
        accelerator="gpu",
        strategy=strat,
        devices=cfg.num_gpus,
        num_nodes=cfg.num_nodes,
        gradient_clip_val=1.0,
        precision=cfg.precision,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        default_root_dir=hydra.utils.get_original_cwd(),
        logger=wandb_logger,
        callbacks=list(callbacks.values()),
        plugins=[SLURMEnvironment(auto_requeue=False)],
        deterministic=True,
    )

    trainer.fit(lightning_module, clip_datamodule)
    print("Training done. Beginning eval suite...")

    # Zero-shot eval on classification datasets
    verbose = torch.cuda.current_device() == 0

    best_model_path = callbacks["checkpoint"].best_model_path
    best_model_folder = os.path.dirname(os.path.dirname(best_model_path))

    mimic_metrics = eval_chexpert(best_model_folder, cfg.data.eval.mimic_path, verbose)
    chexpert_metrics = eval_chexpert(best_model_folder, cfg.data.eval.chexpert_path, verbose)
    cxr14_metrics = eval_cxr14(best_model_folder, cfg.data.eval.cxr14_path, verbose)
    pneumonia_metrics = eval_rsna(best_model_folder, cfg.data.eval.rsna_path, verbose)

    eval_metrics_df = pd.DataFrame(columns=["dataset", "accuracy", "f1"])
    new_rows = [
        {
            "dataset": "MIMIC 5x200",
            "accuracy": mimic_metrics["accuracy"],
            "f1": mimic_metrics["f1"],
        },
        {
            "dataset": "CHEXPERT 5x200",
            "accuracy": chexpert_metrics["accuracy"],
            "f1": chexpert_metrics["f1"],
        },
        {"dataset": "CXR 14x100", "accuracy": cxr14_metrics["accuracy"], "f1": cxr14_metrics["f1"]},
        {
            "dataset": "RSNA Pneumonia",
            "accuracy": pneumonia_metrics["accuracy"],
            "f1": pneumonia_metrics["f1"],
        },
    ]
    eval_metrics_df = pd.concat([eval_metrics_df, pd.DataFrame(new_rows)], ignore_index=True)

    if verbose:
        table = wandb.Table(dataframe=eval_metrics_df)
        print()
        print(eval_metrics_df)
        wandb.log(eval_metrics_df.to_dict())
        wandb.log({"Model Evaluation": table})

        wandb.run.summary["mimic_f1"] = mimic_metrics["f1"]
        wandb.run.summary["chexpert_f1"] = chexpert_metrics["f1"]
        wandb.run.summary["rsna_f1"] = pneumonia_metrics["f1"]
        wandb.finish()


if __name__ == "__main__":
    main()
