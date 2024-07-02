from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from eclip.callbacks import CustomProgressBar


def build_callbacks(cfg: DictConfig):
    callbacks = {}

    custom_progress_bar = CustomProgressBar(total_steps=cfg.max_steps)
    callbacks["pbar"] = custom_progress_bar

    if cfg.callbacks.save_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.model.checkpoint_path,
            monitor="validation_loss",
            mode="min",  # looking for the minimum value of the monitored metric
            save_top_k=1,  # save only the best model
            save_last=False,  # do not save the model from the last epoch
            filename="best_model",  # the filename to use for the .ckpt file
            verbose=True,  # whether to display saving information
        )
        callbacks["checkpoint"] = checkpoint_callback

    if cfg.callbacks.early_stop:
        # Define an EarlyStopping callback
        early_stop_callback = EarlyStopping(
            monitor="validation_loss",
            min_delta=0.00,  # minimum value to determine 'improved'
            patience=cfg.es_patience,  # number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode="min",
        )
        callbacks["early_stop"] = early_stop_callback

    if cfg.callbacks.monitor_lr:
        # log the learning rate at each training step
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks["lr_monitor"] = lr_monitor

    return callbacks
