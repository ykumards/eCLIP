import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datasets import load_dataset
from datasets import load_from_disk

from eclip.constants import CXR8_LABELS


def labels_to_vector_cxr8(labels):
    mapping = {k: v for (v, k) in enumerate(CXR8_LABELS)}
    # Initialize a zero vector of length equal to the number of labels
    vector = [0] * len(mapping)

    # Set the corresponding index to 1 for each label in the instance
    for label_idx in labels:
        vector[label_idx - 1] = 1

    return vector


def labels_to_vector(labels):
    mapping = {
        "No Finding": 0,
        "Atelectasis": 1,
        "Cardiomegaly": 2,
        "Effusion": 3,
        "Infiltration": 4,
        "Mass": 5,
        "Nodule": 6,
        "Pneumonia": 7,
        "Pneumothorax": 8,
        "Consolidation": 9,
        "Edema": 10,
        "Emphysema": 11,
        "Fibrosis": 12,
        "Pleural_Thickening": 13,
        "Hernia": 14,
    }
    # Initialize a zero vector of length equal to the number of labels
    vector = [0] * len(mapping)

    # Set the corresponding index to 1 for each label in the instance
    for label_idx in labels:
        vector[label_idx] = 1

    return vector


class ChestXRay14x100Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset.loc[idx, "image"]
        label_idx = self.dataset.loc[idx, "label"]

        labels = np.zeros(
            14,
        )
        labels[label_idx] = 1

        image = image.convert("RGB")

        augmented = self.transform(image=image)

        return augmented, torch.tensor(labels).float()


class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, transform, split="test"):
        if split not in ["train", "test"]:
            raise ValueError("split must be `train` or `test`")
        self.transform = transform
        self.split = split
        self.image_dir = os.path.join(datadir, f"{split}_images")
        self.df = pd.read_csv(os.path.join(datadir, f"{self.split}_labels.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, f"{row['patientId']}.jpeg")
        image = Image.open(image_path)
        image = image.convert("RGB")

        labels = row["Target"]

        augmented = self.transform(image=image)

        return augmented, torch.tensor(labels).float()


class ChexpertDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, transform, split="test"):
        if split not in ["train", "test"]:
            raise ValueError("split must be `train` or `test`")
        self.transform = transform
        self.split = "valid" if split == "test" else "train"
        self.image_dir = datadir
        self.df = pd.read_csv(os.path.join(datadir, f"{self.split}.csv"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, f"{row['Path']}")
        image = Image.open(image_path)
        image = image.convert("RGB")

        labels = [int(x) for x in eval(row["labels"])]
        augmented = self.transform(image=image)

        return augmented, torch.tensor(labels).float()


class OpeniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        openi_path="/l/Work/aalto/eCLIP/datafiles/open-i-raw",
        transform=None,
        sample_n=None,
        max_length=128,
    ):
        self.tokenizer = tokenizer
        openi_path = Path(openi_path) if not isinstance(openi_path, Path) else openi_path
        self.image_dir = openi_path / "images/images_normalized"

        openi_df = pd.read_csv(openi_path / "indiana_reports.csv")
        openi_projections = pd.read_csv(openi_path / "indiana_projections.csv")
        openi_projections = openi_projections.query("projection == 'Frontal'")
        openi_df = openi_df.merge(openi_projections, on="uid", how="inner")

        if sample_n is not None:
            openi_df = openi_df.sample(n=sample_n, random_state=100)

        self.df = openi_df.dropna().reset_index(drop=True)
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, f"{row['filename']}")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = row["findings"]

        return image, text

    def collate_fn(self, batch):
        images = [x[0] for x in batch]
        texts = [x[1] for x in batch]

        tokenized_text = self.tokenizer(
            texts, max_length=self.max_length, return_tensors="pt", padding=True, truncation=True
        )

        return torch.stack(images), tokenized_text


def get_nih_cxr8_dataset(cache_dir):
    xray_ds_train = load_from_disk(f"{cache_dir}/cxr8_dataset_train")
    split_ds = xray_ds_train.train_test_split(test_size=0.1)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    test_ds = load_from_disk(f"{cache_dir}/cxr8_dataset_test")

    return train_ds, val_ds, test_ds