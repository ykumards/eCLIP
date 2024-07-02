from typing import Any, Dict, Optional, Tuple

import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from eclip.utils.distributed import BackpropType, gather_tensor
from eclip.outputs import ContrastiveLossOutput, GloriaLossOutput


def _gather_embeddings_and_labels(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
) -> Tuple[Tensor, Tensor, Tensor]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        return embeddings_a, embeddings_b, labels

    embeddings_a_all_gpus = gather_tensor(embeddings_a, backprop_type)
    embeddings_b_all_gpus = gather_tensor(embeddings_b, backprop_type)
    # embeddings_a has shape [local_batch_size, embedding_dim]
    local_batch_size = embeddings_a.size(0)
    labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
        local_batch_size, device=embeddings_a.device
    )

    return (
        torch.cat(embeddings_a_all_gpus),
        torch.cat(embeddings_b_all_gpus),
        labels,
    )


class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of input embeddings a and b. For each input_a
    embedding, we compute a weighted cosine similarity with all input_b embeddings,
    then calculate the cross entropy loss against the true (input_a, input_b) pairing.
    Each input_b embedding is evaluated against all input_a embeddings similarly.
    The batch's loss is the average cross entropy over all input_a and input_b embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (Union[float, nn.Module]): Log of the learnable temperature parameter value
            A nn.Parameter instantiation can also be passed directly in case parent class
            is handling the initialization.
            Defaults to ``ln(1/0.07)``, as in the CLIP paper.
        logit_scale_min (Optional[float]): Log of the minimum temperature value.
            If ``None``, then temperature will not be clamped to a minimum value.
            Defaults to ``ln(1)``, as in the CLIP paper.
        logit_scale_max (Optional[float]): Log of the maximum temperature value.
            If ``None``, then temperature will not be clamped to a maximum value.
            Defaults to ``ln(100)``, as in the CLIP paper.

    Inputs: image_embeddings (Tensor): Tensor containing features from the first input or modality.
                (In the CLIP model, these are the outputs of the image encoder.)
            text_embeddings (Tensor): Tensor containing features from the second input or modality.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_type (BackpropType): whether to backpropagate gradients to all
                workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
                Default: BackpropType.GLOBAL
            cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)
            mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
                be considered in the loss calculation use this option to pass a boolean
                mask. Size is (BatchSize,). Defaults to None.
    """

    def __init__(
        self,
        tau: float = 0.07,
        tau_min: Optional[float] = 0.01,
        tau_max: Optional[float] = 2.0,
        fix_temperature=False,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        if not tau_min and not tau_max:
            raise ValueError("Only one of `tau_min` and `tau_max` can be None.")
        # paper says "clipped to prevent scaling the logits by more than 100"
        self.tau_min = tau_min
        self.tau_max = tau_max

        # "learnable temperature parameter τ was initialized to the equivalent of 0.07"
        self.tau = tau * torch.ones([])
        if not fix_temperature:
            self.tau = nn.Parameter(self.tau)

    def forward(
        self,
        image_embeddings: Tensor,
        text_embeddings: Tensor,
        backprop_type: BackpropType = BackpropType.GLOBAL,
    ) -> Tensor:

        self.tau.data.clamp_(self.tau_min, self.tau_max)

        (
            image_embeddings_all_gpus,
            text_embeddings_all_gpus,
            labels,
        ) = _gather_embeddings_and_labels(image_embeddings, text_embeddings, backprop_type)

        # logits_per_image has shape [local_batch_size, global_batch_size]
        logits_per_image = torch.matmul(image_embeddings, text_embeddings_all_gpus.transpose(0, 1)) / self.tau
        logits_per_text = torch.matmul(text_embeddings, image_embeddings_all_gpus.transpose(0, 1)) / self.tau

        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2

        # Calculate accuracy metrics
        a2b_preds = logits_per_image.argmax(dim=1)
        b2a_preds = logits_per_text.argmax(dim=1)

        acc_a2b = (a2b_preds == labels).float().mean()
        acc_b2a = (b2a_preds == labels).float().mean()

        return ContrastiveLossOutput(
            loss=loss,
            logits_image=logits_per_image,
            logits_text=logits_per_text,
            loss_image=loss_image,
            loss_text=loss_text,
            acc_image2text=acc_a2b,
            acc_text2image=acc_b2a,
        )


def sph_inter(a,b,s):
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

class GeodesicClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            unimix=0.0,
            vlmix=0.0,
            mmix=0.0,
            beta_u=0.5,
            beta_m=0.5,
            m_tau=0.01
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        # multimodalmixup
        self.unimix=unimix
        self.vlmix=vlmix
        self.mmix=mmix
        self.m_tau=m_tau
        self.beta_u=beta_u
        self.beta_m=beta_m
        random.seed(1)

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        #! vanilla CL
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        #! -------------------------
        #! CL with multi-modal mixup
        #! -------------------------
        I = torch.eye(image_features.shape[0]).to("cuda:0")
        I_D = 1 - I
        if self.mmix > 0:
            lamb = torch.Tensor([random.betavariate(self.beta_m,self.beta_m)]).to("cuda:0")
            mixed_neg = sph_inter(image_features, text_features, lamb)
            logits_per_image_mm    = self.m_tau * image_features @ mixed_neg.T
            logits_per_text_mm     = self.m_tau * text_features @ mixed_neg.T
            logits_per_image_mm    = logits_per_image*I    +   logits_per_image_mm*I_D
            logits_per_text_mm     = logits_per_text*I     +   logits_per_text_mm*I_D
            mmix_loss = (
                F.cross_entropy(logits_per_image_mm, labels) +
                F.cross_entropy(logits_per_text_mm, labels)
            ) / 2

            total_loss += self.mmix * mmix_loss

        return ContrastiveLossOutput(
            loss=total_loss,
            logits_image=logits_per_image,
            logits_text=logits_per_text,
            acc_image2text=0.0,
            acc_text2image=0.0,
        )

#### OpenCLIP Implementation
def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


###### Gloria Loss Functions
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def gloria_global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    labels = torch.LongTensor(range(batch_size)).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def gloria_local_loss(
    img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = attention_fn(word, context, temp1)  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    labels = torch.LongTensor(range(batch_size)).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps


class GloriaLoss(nn.Module):
    def __init__(
        self,
        temp1,
        temp2,
        temp3,
    ):
        super().__init__()
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.local_loss_weight = 1.0
        self.global_loss_weight = 1.0

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents]
        l_loss0, l_loss1, attn_maps = gloria_local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = gloria_global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def forward(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents):
        l_loss0, l_loss1, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight

        return loss, attn_maps
