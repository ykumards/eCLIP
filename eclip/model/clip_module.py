from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import transformers

from eclip.losses import ContrastiveLossWithTemperature, GeodesicClipLoss
from eclip.utils.distributed import BackpropType


class ClipModule(pl.LightningModule):
    def __init__(
        self,
        config,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            text_encoder.base.config._name_or_path
        )
        self.clip_loss_fn = ContrastiveLossWithTemperature(
            fix_temperature=config.fix_temperature, tau=config.temperature
        )
        self.geodesic_loss_fn = GeodesicClipLoss()
        self.first_batch_processed = False
        self.naive_expert_fuse_method = config.naive_expert_fuse_method

        self.use_geodesic_loss = False
        self.use_dacl = False


    def mixup_data(self, batch, alpha=0.6):
        '''Applies mixup augmentation to a batch of data.'''
        x = batch[0]
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        return (mixed_x, batch[1])

    def _normalize_projections(self, encoder_output):
        encoder_output.projected_vec = F.normalize(
            encoder_output.projected_vec, p=2, dim=-1
        )
        if encoder_output.masked_projected_vec is not None:
            encoder_output.masked_projected_vec = F.normalize(
                encoder_output.masked_projected_vec, p=2, dim=-1
            )
        return encoder_output

    def unpack_batch(self, batch):
        if len(batch) == 2:
            image, text = batch
            heatmap_mask, text_snippet = None, None
        elif len(batch) == 4:
            image, text, heatmap_mask, text_snippet = batch
        else:
            raise ValueError(
                f"Unexpected tuple length for batch: {len(batch)}, it should be 2 or 4"
            )
        # if self.naive_expert_fuse_method:
        #     # naive fusion will ignore the masks even for expert data
        #     # essentially we end up with 1000 more training data for CLIP
        #     # used for ablation
        #     heatmap_mask, text_snippet = None, None
        return image, text, heatmap_mask, text_snippet

    def common_step(self, batch, lambda_=1.0, reconstruct=False) -> torch.Tensor:
        # setting naive_expert_fuse to True will return mask as None
        image, text, heatmap_mask, text_snippet = self.unpack_batch(batch)
        expert_loop: bool = heatmap_mask is not None

        text_encoder_output = self.text_encoder(
            text, snippet=text_snippet, lambda_=lambda_
        )
        text_encoder_output = self._normalize_projections(text_encoder_output)

        image_encoder_output = self.image_encoder(
            image, expert_heatmap=heatmap_mask, lambda_=lambda_, reconstruct=reconstruct
        )
        image_encoder_output = self._normalize_projections(image_encoder_output)

        return image_encoder_output, text_encoder_output

    def training_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        main_batch, expert_batch = batch

        if self.use_dacl:
            mixed_batch_1 = self.mixup_data(main_batch, alpha=0.6)
            mixed_batch_2 = self.mixup_data(main_batch, alpha=0.6)

            image_encoder_output1, text_encoder_output1 = self.common_step(mixed_batch_1)
            image_projection_embed1 = image_encoder_output1.projected_vec
            text_projection_embed1 = text_encoder_output1.projected_vec

            image_encoder_output2, text_encoder_output2 = self.common_step(mixed_batch_2)
            image_projection_embed2 = image_encoder_output2.projected_vec
            text_projection_embed2 = text_encoder_output2.projected_vec

            image_projection_embed = torch.cat([image_projection_embed1, image_projection_embed2])
            text_projection_embed = torch.cat([text_projection_embed1, text_projection_embed2])

        else:
            image_encoder_output, text_encoder_output = self.common_step(main_batch)
            image_projection_embed = image_encoder_output.projected_vec
            text_projection_embed = text_encoder_output.projected_vec

        if self.use_geodesic_loss:
            clip_loss_output = self.geodesic_loss_fn(image_projection_embed, text_projection_embed, 1/0.07)
        else:
            clip_loss_output = self.clip_loss_fn(
                image_embeddings=image_projection_embed,
                text_embeddings=text_projection_embed,
                backprop_type=BackpropType.GLOBAL,
            )

        self.log("training_loss", clip_loss_output.loss, on_step=True)
        self.log(
            "training_imgage2text_acc", clip_loss_output.acc_image2text, on_step=True
        )
        self.log(
            "training_text2image_acc", clip_loss_output.acc_text2image, on_step=True
        )
        self.log(
            "tau", self.clip_loss_fn.tau.item()
        )  # default logit_scale = log 1/0.07
        return clip_loss_output.loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        image_encoder_output, text_encoder_output = self.common_step(batch)
        text_projection_embed = text_encoder_output.projected_vec
        image_projection_embed = image_encoder_output.projected_vec

        clip_loss_output = self.clip_loss_fn(
            image_embeddings=image_projection_embed,
            text_embeddings=text_projection_embed,
            backprop_type=BackpropType.NONE,
        )
        self.log("validation_loss", clip_loss_output.loss, on_step=True, sync_dist=True)
        self.log(
            "validation_imgage2text_acc",
            clip_loss_output.acc_image2text,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "validation_text2image_acc",
            clip_loss_output.acc_text2image,
            on_step=True,
            sync_dist=True,
        )
        return clip_loss_output.loss

    def test_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        return self.validation_step(batch, *args)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_groups, lr=self.config.learning_rate, betas=(0.9, 0.98)
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)

        if self.config.scheduler_name == "cosine":
            hf_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5,
            )
            scheduler = {
                "scheduler": hf_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
        elif self.config.scheduler_name == "plateau":
            torch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=5
            )
            scheduler = {
                "scheduler": torch_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
                "monitor": "validation_loss",
            }
        else:
            raise ValueError(f"Unsupported lr scheduler: {self.config.scheduler_name}")
        return [optimizer], [scheduler]

    def on_epoch_end(self):
        pass

    def _save_first_batch_text(self, batch: Tuple[torch.Tensor, List[str]], tokenizer):
        image, text = batch
        # Save images (as you were already doing)
        torchvision.utils.save_image(image, "first_batch_images.png", nrow=8)
        # Convert token IDs to tokens and then to strings
        with open("first_batch_texts.txt", "w") as file:
            for ids in text["input_ids"]:
                tokens = tokenizer.convert_ids_to_tokens(ids)
                text_str = " ".join(tokens)
                file.write(text_str + "\n")

    def on_train_end(self):
        self.logger.experiment.log(
            {
                "best_model_checkpoint_path": self.trainer.checkpoint_callback.best_model_path
            }
        )
