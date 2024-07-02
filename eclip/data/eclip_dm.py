from pathlib import Path
import PIL.Image as Image
import io
import os
import re
import glob
import json
import random
import pandas as pd
import numpy as np
import torch
import webdataset as wds
import nltk
import scipy
import cv2

from transformers import AutoTokenizer
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from eclip.transforms.image_transforms import (
    CLIPImageTransform,
)
from eclip.utils.common import worker_init_fn


def synchronized_transform(image, heatmap, transform):
    """Apply the same random transformations to both image and heatmap."""

    # Save the current random state
    state = random.getstate()
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()

    # Apply transformations to the image
    transformed_image = transform(image)

    # Restore the saved random state
    random.setstate(state)
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)

    # Apply the same transformations to the heatmap
    transformed_heatmap = transform(heatmap, is_mask=True)

    return transformed_image, transformed_heatmap


class ExpertClipDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_path = cfg.data.data_path
        self.expert_data_path = cfg.data.expert_data_path
        self.text_model_name = cfg.model.text.model_name
        self.train_epoch_size = cfg.data.train_data_size
        self.val_epoch_size = cfg.data.val_data_size
        self.batch_size = cfg.batch_size
        self.img_size = cfg.data.img_size
        self.num_workers = cfg.num_workers
        self.max_length = cfg.max_length
        self.seed = cfg.seed
        self.clip_post_tune = cfg.clip_post_tune

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.train_transform = CLIPImageTransform(
            image_size=self.img_size,
            image_mean=cfg.data.mean,
            image_std=cfg.data.std,
            is_train=True,
        )
        self.val_transform = CLIPImageTransform(
            image_size=self.img_size,
            image_mean=cfg.data.mean,
            image_std=cfg.data.std,
            is_train=False,
        )
        self.use_main_loader = True

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_datasetdir = os.path.join(self.data_path, "train")
            train_dataset_urls = sorted(glob.glob(os.path.join(train_datasetdir, "*.tar")))

            self.train_dataset = wds.DataPipeline(
                wds.ResampledShards(train_dataset_urls, seed=self.seed, deterministic=True),
                wds.detshuffle(1000),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.map(self.decode_image),
                wds.to_tuple("frontal_img", "report"),
                wds.map_tuple(self.train_transform, self.decode_text),
                wds.batched(self.batch_size),
                wds.map(self.process_text_batch),
            ).with_epoch(self.train_epoch_size)

            expert_dataset_urls = sorted(glob.glob(os.path.join(self.expert_data_path, "*.tar")))
            self.expert_dataset = wds.DataPipeline(
                wds.ResampledShards(expert_dataset_urls, seed=self.seed, deterministic=True),
                wds.detshuffle(300),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode("pil"),
                wds.map(self.decode_expert_image),
                wds.map(self.decode_heatmaps),
                wds.map(self.extract_heatmap_snippet_pair),
                wds.map(self.joint_image_transform),
                wds.map(
                    lambda sample: (
                        sample["img.jpg"],
                        sample["report.txt"],
                        sample["heatmap"],
                        sample["text_snippet"],
                    )
                ),
                wds.batched(self.batch_size//2), # contrastive loss gets oom
                wds.map(self.process_expert_batch),
            ).with_epoch(1000)

        if stage == "validate" or stage is None:
            val_datasetdir = os.path.join(self.data_path, "val")
            val_dataset_urls = sorted(glob.glob(os.path.join(val_datasetdir, "*.tar")))

            self.val_dataset = wds.DataPipeline(
                wds.ResampledShards(val_dataset_urls, seed=self.seed, deterministic=True),
                wds.detshuffle(1000),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.map(self.decode_image),
                wds.to_tuple("frontal_img", "report"),
                wds.map_tuple(self.train_transform, self.decode_text),
                wds.batched(self.batch_size),
                wds.map(self.process_text_batch),
            ).with_epoch(self.val_epoch_size)

        print("Datamodule setup done.")

    def train_dataloader(self):
        return [self.get_main_loader(), self.get_expert_loader()]

    def get_main_loader(self):
        return wds.WebLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def get_expert_loader(self):
        return wds.WebLoader(
            self.expert_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=None,
            pin_memory=True,
        )

    def decode_expert_image(self, sample):
        sample["img.jpg"] = sample["img.jpg"].convert("RGB")
        return sample

    def decode_heatmaps(self, sample):
        """Decode heatmaps from the sample."""
        heatmaps = []
        text_snippets = []
        if "manifest.json" in sample and len(sample["manifest.json"]) > 0:
            for item in sample["manifest.json"]:
                heatmap_key = item["heatmap"]
                snippet_key = item["snippet"]
                if heatmap_key in sample and snippet_key in sample:
                    components = sample[heatmap_key]
                    shape = components["shape"]
                    indices = components["indices"]
                    indptr = components["indptr"]
                    data = components["data"]
                    heatmap_sparse = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
                    heatmap = heatmap_sparse.toarray()

                    heatmaps.append(heatmap)
                    text_snippets.append(sample[snippet_key])

            sample["heatmaps"] = heatmaps
            sample["text_snippets"] = text_snippets
        else:
            img_size = sample["img.jpg"].size
            sample["heatmaps"] = [Image.fromarray(np.ones(img_size))]
            # when there's no heatmap, we use the whole report
            sample["text_snippets"] = [sample["report.txt"]]

        return sample

    def extract_heatmap_snippet_pair(self, sample):
        """Pick a random heatmap-snippet pair"""
        heatmps = Image.fromarray(sample["heatmaps"][0].astype(np.uint8))
        snips = ".".join(sample["text_snippets"])
        sample["heatmap"] = heatmps
        sample["text_snippet"] = snips

        sample.pop("heatmaps")
        sample.pop("text_snippets")
        sample.pop("manifest.json")
        return sample

    def joint_image_transform(self, sample):
        image = sample["img.jpg"]
        heatmap = sample["heatmap"]

        if heatmap is not None:
            # Apply synchronized transformations
            image, heatmap = synchronized_transform(image, heatmap, self.train_transform)
        else:
            image = self.train_transform(image)

        sample["img.jpg"] = image
        sample["heatmap"] = heatmap
        return sample

    def process_expert_batch(self, batch):
        images, texts, heatmaps, text_snippets = batch
        tokenized_texts = self.tokenizer(
            texts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True
        )
        tokenized_text_snippets = self.tokenizer(
            text_snippets,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return images, tokenized_texts, heatmaps, tokenized_text_snippets

    def decode_image(self, sample):
        if "frontal_img" in sample:
            sample["frontal_img"] = Image.open(io.BytesIO(sample["frontal_img"])).convert("RGB")
        return sample

    def decode_text(self, report):
        return report.decode("utf-8")

    def process_text_batch(self, batch):
        images, texts = batch

        tokenized_texts = self.tokenizer(
            texts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True
        )

        return images, tokenized_texts
