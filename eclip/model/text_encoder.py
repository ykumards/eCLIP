import torch.nn as nn
import transformers

from eclip.model.blocks import (
    mean_pooling_with_mask,
    max_pooling_with_mask,
    ExpertMixupModule,
)
from eclip.outputs import EncoderOutput


class TextEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        base: nn.Module,
        projector: nn.Module,
    ):
        super().__init__()

        self.base = base
        self.projector = projector
        self.embed_method = cfg.model.text.embed_method

    def forward(self, x, snippet=None, lambda_=1.0):
        out_last = self.base(**x).last_hidden_state

        if self.embed_method == "cls":
            out = out_last[:, 0, :]
        elif self.embed_method == "mean":
            out = mean_pooling_with_mask(out_last, x["attention_mask"])
        elif self.embed_method == "max":
            out = max_pooling_with_mask(out_last, x["attention_mask"])
        else:
            raise ValueError(
                f"Supported embed methods are 'cls' and 'mean', {self.embed_method} was passed"
            )
        projected_vec = self.projector(out)
        return EncoderOutput(projected_vec=projected_vec)


class ExpertTextEncoder(nn.Module):
    def __init__(
        self,
        cfg,
        base: nn.Module,
        projector: nn.Module,
    ):
        super().__init__()

        self.config = cfg
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base.config._name_or_path
        )

        self.base = base
        self.projector = projector
        self.embed_method = cfg.model.text.embed_method
        self.mixup_module = ExpertMixupModule(mixup_type=cfg.model.mixup_type)

        self.n_layers = len(self.base.encoder.layer)
        self.n_inter_layers = 3
        assert (
            self.n_inter_layers < self.n_layers
        ), f"intermediate layers ({self.n_inter_layers}) should be less than total layers ({self.n_layers})"

    def forward(self, x, snippet=None, lambda_=1.0):
        out_last = self.base(**x).last_hidden_state
        if self.embed_method == "cls":
            out = out_last[:, 0, :]
        elif self.embed_method == "mean":
            out = mean_pooling_with_mask(out_last, x["attention_mask"])
        elif self.embed_method == "max":
            out = max_pooling_with_mask(out_last, x["attention_mask"])
        else:
            raise ValueError(
                f"Supported embed methods are 'cls' and 'mean', {self.embed_method} was passed"
            )

        projected_vec = self.projector(out)

        # this is for future work, we don't handle the text snippet for now
        snippet_projected_vec = None
        return EncoderOutput(
            projected_vec=projected_vec,
            masked_projected_vec=snippet_projected_vec
        )
