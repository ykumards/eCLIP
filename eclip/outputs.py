from dataclasses import dataclass, field
from typing import OrderedDict, List

from torch import Tensor


@dataclass
class EncoderOutput(OrderedDict):
    projected_vec: Tensor = None
    masked_projected_vec: Tensor = None
    attention_map: Tensor = None
    reconstruction_loss: Tensor = None
    mask_loss: Tensor = None
    mask_vicinal_loss: Tensor = None
    embed: Tensor = None
    masked_embed: Tensor = None


@dataclass
class GloriaEncoderOutput(EncoderOutput):
    local_embed: Tensor = None
    masked_local_embed: Tensor = None
    sents: List = field(default_factory=list)


@dataclass
class ContrastiveLossOutput(OrderedDict):
    loss: Tensor = None
    logits_image: Tensor = None
    logits_text: Tensor = None
    loss_image: Tensor = None
    loss_text: Tensor = None
    acc_image2text: Tensor = None
    acc_text2image: Tensor = None


@dataclass
class GloriaLossOutput(OrderedDict):
    global_loss: Tensor
    local_loss: Tensor
    logits_image: Tensor
    logits_text: Tensor
    loss_image: Tensor
    loss_text: Tensor
    acc_image2text: Tensor
    acc_text2image: Tensor
