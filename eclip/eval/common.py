from pathlib import Path

from eclip.model.clip_module import ClipModule
from eclip.model.eclip_module import ExpertClipModule
from eclip.model.gloria_module import GloriaModule
from eclip.model.expert_gloria_module import ExpertGloriaModule


def load_pl_module(checkpoint_path, use_expert=False, load_gloria=False):
    if load_gloria:
        module_cls = ExpertGloriaModule if use_expert else GloriaModule
    else:
        module_cls = ExpertClipModule if use_expert else ClipModule
    checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
    lightning_module = module_cls.load_from_checkpoint(
        checkpoint_path / "saved_model/best_model.ckpt", strict=False
    )
    print(f"Loaded module: {module_cls}")
    return lightning_module
