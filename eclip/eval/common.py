from pathlib import Path

from eclip.model.clip_module import ClipModule
from eclip.model.eclip_module import ExpertClipModule


def load_pl_module(checkpoint_path, use_expert=False):
    module_cls = ExpertClipModule if use_expert else ClipModule
    checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
    lightning_module = module_cls.load_from_checkpoint(
        checkpoint_path / "saved_model/best_model.ckpt", strict=False
    )
    print(f"Loaded module: {module_cls}")
    return lightning_module
