from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import transformers
import numpy as np

from eclip.model.clip_module import ClipModule
from eclip.model.blocks import ExpertMixupModule
from eclip.utils.distributed import BackpropType


class ExpertClipModule(ClipModule):
    def __init__(
        self,
        config,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
    ) -> None:
        super().__init__(config, image_encoder, text_encoder)
        self.save_hyperparameters()

        self.mixup_module = ExpertMixupModule(mixup_type=config.model.mixup_type)

        self.mixup_alpha = config.model.mixup_alpha
        self.aux_loss_proportion = config.aux_loss_proportion
        self.xent_temperature = config.xent_temperature
        self.expert_min_prob = config.expert_min_prob
        self.expert_max_prob = config.expert_max_prob
        self.expert_cold_start_ratio = config.expert_cold_start_ratio
        self.expert_warmup_ratio = config.expert_warmup_ratio
        self.expert_cooldown_ratio = config.expert_cooldown_ratio
        self.naive_expert_fuse_method = config.naive_expert_fuse_method

        self.expert_cold_start_steps = int(
            config.max_steps * self.expert_cold_start_ratio
        )
        self.expert_warmup_steps = int(config.max_steps * self.expert_warmup_ratio)
        self.expert_cooldown_steps = int(config.max_steps * self.expert_cooldown_ratio)

        assert (
            self.expert_cold_start_steps
            + self.expert_warmup_steps
            + self.expert_cooldown_steps
        ) <= config.max_steps, (
            "expert prob ratio is set incorrectly, they should add up to 1.0"
        )

        self.expert_global_step = 0

    def _get_current_prob(
        self,
        iteration,
        cooling_steps,
        warmup_steps,
        anneal_steps,
        start_prob=0.1,
        max_prob=0.8,
    ):
        if iteration <= cooling_steps:
            # Cooling stage
            prob = 0.0
        elif iteration <= cooling_steps + warmup_steps:
            # Linear warm-up
            prob = start_prob + (max_prob - start_prob) * (
                (iteration - cooling_steps) / warmup_steps
            )
        elif iteration <= cooling_steps + warmup_steps + anneal_steps:
            # Linear anneal
            prob = max_prob - (max_prob - start_prob) * (
                (iteration - cooling_steps - warmup_steps) / anneal_steps
            )
        else:
            # After annealing
            prob = start_prob
        return prob

    def training_step(self, batch, *args: list) -> torch.Tensor:
        # pick a batch from the dataloaders
        # we get expert_batch and then main_batch
        main_batch, expert_batch = batch

        expert_prob = self._get_current_prob(
            iteration=self.global_step,
            start_prob=self.expert_min_prob,
            max_prob=self.expert_max_prob,
            cooling_steps=self.expert_cold_start_steps,
            warmup_steps=self.expert_warmup_steps,
            anneal_steps=self.expert_cooldown_steps,
        )
        if self.naive_expert_fuse_method:
            expert_prob = 1.0 # always use v^E

        reconstruct = self.global_step < self.expert_cold_start_steps
        image_encoder_output, text_encoder_output = self.common_step(main_batch, reconstruct=reconstruct)

        image_embeds = image_encoder_output.projected_vec
        text_embeds = text_encoder_output.projected_vec

        aux_loss = image_encoder_output.reconstruction_loss
        if np.random.rand() < expert_prob:
            self.expert_global_step += 1

            lambda_ = np.random.beta(a=self.mixup_alpha, b=self.mixup_alpha)

            if self.naive_expert_fuse_method:
                lambda_ = 0.0 # always use v^E

            # process and inject expert batch
            expert_image_encoder_output, expert_text_encoder_output = self.common_step(expert_batch, lambda_=lambda_)

            image_embeds = torch.cat([image_embeds, expert_image_encoder_output.masked_projected_vec])

            # either use the snippet projection or the original report for expert batch
            text_embeds = torch.cat([text_embeds, expert_text_encoder_output.projected_vec])


        main_clip_output_loss = self.clip_loss_fn(
            image_embeddings=image_embeds,
            text_embeddings=text_embeds,
            backprop_type=BackpropType.GLOBAL,
        )

        self.log("clip_training_loss", main_clip_output_loss.loss, on_step=True)
        self.log(
            "training_image2text_acc",
            main_clip_output_loss.acc_image2text,
            on_step=True,
        )
        self.log(
            "training_text2image_acc",
            main_clip_output_loss.acc_text2image,
            on_step=True,
        )
        self.log(
            "tau", self.clip_loss_fn.tau.item()
        )  # default logit_scale = log 1/0.07
        self.log("experts_processed", self.expert_global_step)
        self.log("expert_prob", expert_prob)
        total_loss = main_clip_output_loss.loss
        if aux_loss is not None:
            total_loss = (
                1 - self.aux_loss_proportion
            ) * total_loss + self.aux_loss_proportion * aux_loss
            self.log("expert/auxillary_loss", aux_loss.item(), on_step=True)
        self.log("training_loss", total_loss, on_step=True)
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        decoder_lr_multiplier = 2
        base_image_param_group = [
            {
                "params": [
                    p
                    for n, p in self.image_encoder.named_parameters()
                    if not any(
                        nd in n
                        for nd in ["adaptive_concat", "bias", "LayerNorm.weight", "BatchNorm"]
                    )
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.image_encoder.named_parameters()
                    if not any(nd in n for nd in ["adaptive_concat"])
                    and any(nd in n for nd in ["bias", "LayerNorm.weight", "BatchNorm"])
                ],
                "weight_decay": 0.0,  # No weight decay for biases and LayerNorm weights
                "lr": self.config.learning_rate,
            },
        ]
        decoder_param_group = [
            {
                "params": [
                    p
                    for n, p in self.image_encoder.adaptive_concat.named_parameters()
                    if not any(
                        nd in n for nd in ["bias", "LayerNorm.weight", "BatchNorm"]
                    )
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate * decoder_lr_multiplier,  # Higher learning rate for the decoder
            },
            {
                "params": [
                    p
                    for n, p in self.image_encoder.adaptive_concat.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight", "BatchNorm"])
                ],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate * decoder_lr_multiplier,
            },
        ]
        text_param_groups = [
            {
                "params": [
                    p
                    for n, p in self.text_encoder.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight", "BatchNorm"])
                ],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.text_encoder.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight", "BatchNorm"])
                ],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate,
            },
        ]
        param_groups = base_image_param_group + text_param_groups + decoder_param_group
        if self.clip_loss_fn.tau.requires_grad:
            param_groups += [{"params": self.clip_loss_fn.parameters(), "weight_decay": 0.0, "lr": self.config.learning_rate}]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98))
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
