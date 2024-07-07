from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectorConfig:
    input_dim: int = 2048
    output_dim: int = 512
    num_proj_layers: int = 3
    bias: bool = False


class ProjectionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation=nn.GELU(),
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.5)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.dropout(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class ProjectionBlockConVirt(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation=nn.ReLU(),
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(self.activation(x))
        return x


class ProjectionBlockSimple(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation=None) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.Identity()
        if activation is not None:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class Projector(nn.Module):
    def __init__(self, proj_cfg: ProjectorConfig, proj_block_cls) -> None:
        super().__init__()

        self.cfg = proj_cfg

        layers = [proj_block_cls(self.cfg.input_dim, self.cfg.output_dim)]
        for _ in range(self.cfg.num_proj_layers - 1):
            layers.append(proj_block_cls(self.cfg.output_dim, self.cfg.output_dim))
        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        # we will do the norm manually everywhere
        return x


def mean_pooling_with_mask(
    text_representation: torch.FloatTensor, attention_mask: torch.LongTensor
) -> torch.FloatTensor:
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(text_representation.size()).float()
    )
    return torch.sum(text_representation * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def max_pooling_with_mask(
    text_representation: torch.FloatTensor, attention_mask: torch.LongTensor
) -> torch.FloatTensor:
    # Expanding the attention mask for the size of text_representation
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        text_representation.size()
    )

    # Assign a very small value for masked elements so they don't affect the max operation
    text_representation[input_mask_expanded == 0] = -1e9

    # Perform max pooling
    max_pooled = torch.max(text_representation, dim=1)[0]
    return max_pooled


class ExpertMixupModule(nn.Module):
    """Main model that applies the transformed mask to an image."""

    def __init__(self, mixup_type="linear"):
        super(ExpertMixupModule, self).__init__()
        if mixup_type not in ["linear", "random", "identity"]:
            raise ValueError(f"Mixup type: {mixup_type} not supported")
        self.mixup_type = mixup_type

    def forward_mixup_linear(self, x, masked_x, lambda_=1.0):
        # Mixup strategy: combine masked and original x
        mixed_x = lambda_ * x + (1 - lambda_) * masked_x
        return mixed_x

    def forward_random(self, x, masked_x, lambda_=1.0):
        bs = x.shape[0]
        random_noise = torch.normal(mean=torch.zeros(bs), std=0.01 * torch.ones(bs))
        mixed_x = random_noise * x
        return mixed_x

    def forward(self, x, masked_x, lambda_=1.0):
        if self.mixup_type == "linear":
            return self.forward_mixup_linear(x, masked_x, lambda_=lambda_)
        elif self.mixup_type == "random":
            return self.forward_random(x, masked_x)
        # else identity
        return x


class HeatmapAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, heatmap_patches):
        x_transposed = x.transpose(0, 1)
        heatmap_patches_transposed = heatmap_patches.transpose(0, 1)

        query = heatmap_patches_transposed # + x_transposed

        attn_output, attn_map  = self.multihead_attn(query=query, key=x_transposed, value=x_transposed)

        attn_output = attn_output.transpose(0, 1)
        output = self.out_proj(attn_output)
        return output


class AlphaMaskEncoderMHAWithHeatmap(nn.Module):
    def __init__(self, img_dim=224, patch_size=16, embed_dim=256, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patchify = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.attn = HeatmapAttention(embed_dim, num_heads)
        self.unpatchify = nn.ConvTranspose2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, expert_heatmap):
        # Concatenate image and heatmap before patchification
        # Patchify
        patches = self.patchify(x)  # (B, embed_dim, P, P)
        B, C, H, W = patches.size()
        patches = patches.permute(0, 2, 3, 1).reshape(B, H*W, C)  # Reshape to (B, N, embed_dim)

        # Apply custom attention with heatmap influence
        heatmap_patches = self.patchify(expert_heatmap * x)

        _, C_hm, H, W = heatmap_patches.size()
        heatmap_patches = heatmap_patches.permute(0, 2, 3, 1).reshape(B, H*W, C_hm)

        transformed_patches = self.attn(patches, heatmap_patches)

        # Reassemble image (simplified for demonstration, needs to match exact reverse operation)
        transformed_patches = transformed_patches.permute(0, 2, 1).view(B, C, H, W)
        reconstructed_image = self.unpatchify(transformed_patches)

        return reconstructed_image


_projector_class_register = {
    "simple": ProjectionBlockSimple,
    "convirt": ProjectionBlockConVirt,
    "gloria": ProjectionBlockSimple,
}


def make_image_projector(cfg, img_input_dim):
    image_projector_config = ProjectorConfig(
        input_dim=img_input_dim,
        output_dim=cfg.model.proj_dim,
        num_proj_layers=cfg.model.image.num_proj_layers,
        bias=cfg.model.image.proj_bias,
    )
    image_projector = Projector(
        proj_cfg=image_projector_config,
        proj_block_cls=_projector_class_register[cfg.model.proj_type],
    )

    return image_projector


def make_text_projector(cfg, text_input_dim):
    text_projector_config = ProjectorConfig(
        input_dim=text_input_dim,
        output_dim=cfg.model.proj_dim,
        num_proj_layers=cfg.model.text.num_proj_layers,
        bias=cfg.model.text.proj_bias,
    )
    text_projector = Projector(
        proj_cfg=text_projector_config,
        proj_block_cls=_projector_class_register[cfg.model.proj_type],
    )

    return text_projector
