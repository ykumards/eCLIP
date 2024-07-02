from omegaconf import DictConfig

from eclip.data import ExpertClipDataModule


def build_datamodule(cfg: DictConfig):
    use_expert = cfg.use_expert
    _module_mapping = {
        "clip": ExpertClipDataModule,
        "convirt": ExpertClipDataModule,
        "gloria": ExpertClipDataModule,
    }
    _expert_module_mapping = {
        "clip": ExpertClipDataModule,
        "convirt": ExpertClipDataModule,
        "gloria": ExpertClipDataModule,
    }
    mapping = _expert_module_mapping if use_expert else _module_mapping
    clip_datamodule = mapping[cfg.model.clip_model_type](cfg)
    clip_datamodule.setup()
    return clip_datamodule
