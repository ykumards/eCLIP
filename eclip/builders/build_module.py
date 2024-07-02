from omegaconf import DictConfig
import timm
import torch
from transformers import AutoModel


from eclip.model.clip_module import ClipModule
from eclip.model.eclip_module import ExpertClipModule
from eclip.model.image_encoder import (
    SwinEncoder,
    VitTinyEncoder,
    VitSmallEncoder,
    VitBaseEncoder,
    ExpertVitTinyEncoder,
    ExpertVitSmallEncoder,
    ExpertVitBaseEncoder,
    ExpertSwinEncoder,
)
from eclip.model.text_encoder import TextEncoder, ExpertTextEncoder
from eclip.model.blocks import make_image_projector, make_text_projector


_vision_encoder_mapping = {
    "vit-tiny": VitTinyEncoder,
    "vit-small": VitSmallEncoder,
    "vit-base": VitBaseEncoder,
    "swin": SwinEncoder,
}

_vision_expert_encoder_mapping = {
    "vit-tiny": ExpertVitTinyEncoder,
    "vit-small": ExpertVitSmallEncoder,
    "vit-base": ExpertVitBaseEncoder,
    "swin": ExpertSwinEncoder,
}


def _vision_inputdim_mapping(model_name, image_base):
    if model_name == "resnet50":
        raise NotImplementedError
    elif model_name in ["vit-small", "vit-base", "vit-tiny"]:
        return image_base.head.in_features
    elif model_name == "swin":
        return image_base.head.fc.in_features
    else:
        raise ValueError(f"model_name == {model_name} if not supported")


def build_text_encoder(cfg: DictConfig):
    text_base = AutoModel.from_pretrained(cfg.model.text.model_name)
    projector = make_text_projector(cfg, text_base.config.hidden_size)
    encoder_cls = ExpertTextEncoder if cfg.use_expert else TextEncoder
    text_encoder = encoder_cls(cfg=cfg, base=text_base, projector=projector)
    return text_encoder


def build_image_encoder(cfg: DictConfig):
    encoder_mapping = _vision_expert_encoder_mapping if cfg.use_expert else _vision_encoder_mapping
    img_encoder_cls = encoder_mapping[cfg.model.image.model_name]

    image_base = timm.create_model(
        img_encoder_cls.hf_modelname,
        pretrained=cfg.model.pretrained,
    )
    img_input_dim = _vision_inputdim_mapping(cfg.model.image.model_name, image_base)

    projector = make_image_projector(cfg, img_input_dim)
    image_encoder = img_encoder_cls(cfg=cfg, base=image_base, projector=projector)
    return image_encoder


def build_clipmodule(cfg: DictConfig):
    text_encoder = build_text_encoder(cfg)
    image_encoder = build_image_encoder(cfg)

    lightning_module = ClipModule(
        config=cfg,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )

    return lightning_module


def build_expert_clipmodule(cfg: DictConfig):
    text_encoder = build_text_encoder(cfg)
    image_encoder = build_image_encoder(cfg)

    lightning_module = ExpertClipModule(
        config=cfg,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    )
    return lightning_module


def build_eclip_module(cfg: DictConfig):
    text_encoder = build_text_encoder(cfg)
    image_encoder = build_image_encoder(cfg)

    lightning_module = ExpertClipModule(
        config=cfg, image_encoder=image_encoder, text_encoder=text_encoder
    )
    return lightning_module


def build_module(cfg: DictConfig):
    use_expert = cfg.use_expert
    _module_mapping = {
        "clip": build_clipmodule,
        "eclip": build_eclip_module,
    }
    _expert_module_mapping = {
        "clip": build_expert_clipmodule,
        "eclip": build_eclip_module,
    }
    mapping = _expert_module_mapping if use_expert else _module_mapping
    return mapping[cfg.model.clip_model_type](cfg)
