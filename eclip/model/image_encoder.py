import torch
import torch.nn as nn
import torch.nn.functional as F


from eclip.model.blocks import (
    ExpertMixupModule,
    ProjectionBlockSimple,
    Projector,
    AlphaMaskEncoderMHAWithHeatmap
)
from eclip.outputs import EncoderOutput


class EncoderBase(nn.Module):
    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__()
        self.base = base
        self.projector = projector

    def forward(self, x, expert_heatmap=None, lambda_=1.0, reconstruct=False):
        image_embed = self.base(x)
        projected_vec = self.projector(image_embed)
        return EncoderOutput(projected_vec=projected_vec, embed=image_embed)


class VitEncoder(EncoderBase):
    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__(cfg, base, projector)
        self.base.head = nn.Identity()


class VitTinyEncoder(VitEncoder):
    hf_modelname = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"


class VitSmallEncoder(VitEncoder):
    hf_modelname = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"


class VitBaseEncoder(VitEncoder):
    hf_modelname = "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"


class SwinEncoder(EncoderBase):
    hf_modelname = "timm/swin_tiny_patch4_window7_224.ms_in1k"
    # hf_modelname = "timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k"

    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__(cfg, base, projector)
        self.base.head.fc = nn.Identity()


### Expert Encoders
class ExpertEncoderBase(nn.Module):
    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__()
        self.config = cfg
        self.base = base

        self.projector = projector
        proj_config = projector.cfg
        self.mask_projector = Projector(proj_config, ProjectionBlockSimple)
        self.adaptive_concat = AlphaMaskEncoderMHAWithHeatmap()
        self.counter = 0

        self.mixup_module = ExpertMixupModule(mixup_type=cfg.model.mixup_type)

    def mse_loss_fn(self, original_image, reconstructed_image, reduction="mean"):
        """Simple MSE version"""
        return F.mse_loss(original_image, reconstructed_image, reduction=reduction)


class ExpertVitEncoder(ExpertEncoderBase):
    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__(cfg, base, projector)
        self.base.head = nn.Identity()
        self.adaptive_concat = AlphaMaskEncoderMHAWithHeatmap()

    def forward(self, image, expert_heatmap=None, lambda_=1.0, reconstruct=False):
        """
        if mask is set to None, we assume this is not expert batch
        """
        self.counter += 1

        mse_loss = None
        if reconstruct:
            # heatmap_mask will be ones for general image
            reconc_image = self.adaptive_concat(image, torch.ones_like(image)[:, 0:1, :, :])
            mse_loss = self.mse_loss_fn(image, reconc_image)

        features = self.base.forward_features(image)
        image_embed = self.base.forward_head(features)
        image_projected_vec = self.projector(image_embed)

        heatmap_image_projected_vec = None
        expert_image_embed = None
        if expert_heatmap is not None:
            expert_heatmap = expert_heatmap / expert_heatmap.max()
            expert_image = self.adaptive_concat(image, expert_heatmap)

            expert_image = self.mixup_module(image, expert_image, lambda_=lambda_)

            expert_features = self.base.forward_features(expert_image)
            expert_image_embed = self.base.forward_head(expert_features)
            heatmap_image_projected_vec = self.projector(expert_image_embed)

        return EncoderOutput(
            projected_vec=image_projected_vec,
            masked_projected_vec=heatmap_image_projected_vec,
            embed=expert_image_embed,
            masked_embed=expert_image_embed,
            reconstruction_loss=mse_loss,
        )


class ExpertVitTinyEncoder(ExpertVitEncoder):
    hf_modelname = "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k"


class ExpertVitSmallEncoder(ExpertVitEncoder):
    hf_modelname = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"


class ExpertVitBaseEncoder(ExpertVitEncoder):
    hf_modelname = "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k"


class ExpertSwinEncoder(ExpertVitEncoder):
    hf_modelname = "timm/swin_tiny_patch4_window7_224.ms_in1k"

    def __init__(self, cfg, base: nn.Module, projector: nn.Module):
        super().__init__(cfg, base, projector)
        self.base.head.fc = nn.Identity()
        self.adaptive_concat = AlphaMaskEncoderMHAWithHeatmap()
