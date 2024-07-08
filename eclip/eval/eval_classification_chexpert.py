import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from itertools import product
from rich.console import Console
from rich.table import Table


import eclip.constants as const
from eclip.model.clip_module import ClipModule

# from eclip.data.openi_datamodule import OpeniDataModule
from eclip.eval.common import load_pl_module
from eclip.utils import load_config, manual_seed_everything
from eclip.transforms.image_transforms import CLIPImageTransform
from eclip.data import ChexpertDataset


def get_chexpert_prompts_ens(tokenizer):
    prompts = []
    for i, (disease, attr) in enumerate(const.CHEXPERT_CLASS_PROMPTS.items()):
        severity = attr["severity"]
        subtype = attr["subtype"]
        location = attr["location"]
        combinations = [
            f"{s} {sub} {loc}".strip()
            for s, sub, loc in product(severity, subtype, location)
            if s or sub or loc  # Ensure at least one of the elements is non-empty
        ]
        tokenized_combinations = tokenizer(
            combinations,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompts.append(tokenized_combinations)
    return prompts


def evaluate_testset(model, text_embeddings, test_loader, device, verbose=False):
    text_embeddings = text_embeddings.to(device)
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    iterable = tqdm(test_loader, desc="Evaluating", leave=False) if verbose else test_loader
    with torch.no_grad():
        for batch in iterable:
            images, labels = batch
            images = images.to(device)

            # Get image embeddings
            image_embed_output = model.image_encoder(images)
            image_embeddings = image_embed_output.projected_vec
            normed_image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

            # Compute similarity scores for all images in the batch
            similarity_scores = normed_image_embeddings @ text_embeddings.T

            predicted_labels = np.argmax(similarity_scores.cpu().numpy(), axis=1)

            all_preds.append(predicted_labels)
            all_labels.append(labels.numpy())

    # Concatenate all batch results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_labels, all_preds


def get_chexpert_metrics(all_labels, all_preds):
    converted_preds = np.eye(all_labels.shape[1])[all_preds]

    accuracy = accuracy_score(all_labels, converted_preds)
    f1 = f1_score(all_labels, converted_preds, average="macro")

    return {"accuracy": accuracy, "f1": f1}


def eval_chexpert(
    path_to_training_output_folder, chexpert_datadir, verbose=False
):
    path_to_training_output_folder = (
        Path(path_to_training_output_folder)
        if not isinstance(path_to_training_output_folder, Path)
        else Path
    )
    chexpert_datadir = (
        Path(chexpert_datadir) if not isinstance(chexpert_datadir, Path) else chexpert_datadir
    )

    cfg = load_config(path_to_training_output_folder / ".hydra/config.yaml")
    clip_module = load_pl_module(
        path_to_training_output_folder, use_expert=cfg.use_expert
    )
    clip_module.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    clip_module = clip_module.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_name)
    img_transform = CLIPImageTransform(
        cfg.data.img_size, image_mean=cfg.data.mean, image_std=cfg.data.std, is_train=False
    )

    test_dataset = ChexpertDataset(
        datadir=chexpert_datadir,
        transform=img_transform,
        split="train",  # zero-shot learning so we get more samples from here
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=4
    )

    chexpert_prompts = get_chexpert_prompts_ens(tokenizer=tokenizer)
    chexpert_prompts = [{k: v.to(device) for k, v in prompt.items()} for prompt in chexpert_prompts]

    normed_chexpert_embeddings_ensemble = []
    with torch.no_grad():
        for prompt in chexpert_prompts:
            text_encoder_output = clip_module.text_encoder(prompt)
            text_embeddings = text_encoder_output.projected_vec
            normed_text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
            normed_chexpert_embeddings_ensemble.append(normed_text_embeddings)

    normed_chexpert_embeddings = [
        disease_embeds.mean(dim=0) for disease_embeds in normed_chexpert_embeddings_ensemble
    ]
    normed_chexpert_embeddings = torch.stack(normed_chexpert_embeddings, dim=0)
    all_labels, all_preds = evaluate_testset(
        clip_module, normed_chexpert_embeddings, test_loader, device="cuda", verbose=verbose
    )
    test_metrics = get_chexpert_metrics(all_labels, all_preds)

    return test_metrics


if __name__ == "__main__":
    manual_seed_everything(2024)

    chexpert_datadir = Path("")
    mimic_datadir = Path("")

    model_paths = [
        ("CLIP", ""),
        ("eCLIP", "")

    ]

    # Eval on Chexpert
    table = Table(title="Zero-shot Classification \n (Chexpert 5x200)")
    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Acc", justify="right", style="magenta")
    table.add_column("F1 Score", justify="right", style="magenta")

    for model_name, clip_model_checkpoint_dir in model_paths:
        test_metrics = eval_chexpert(clip_model_checkpoint_dir, chexpert_datadir, verbose=True)
        acc = test_metrics['accuracy']
        f1 = test_metrics['f1']

        table.add_row(model_name, f"{acc:.4f}", f"{f1:.4f}")

    console = Console()
    console.print(table)

    # Eval on MIMIC
    table = Table(title="Zero-shot Classification \n (MIMIC 5x200)")
    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Acc", justify="right", style="magenta")
    table.add_column("F1 Score", justify="right", style="magenta")

    for model_name, clip_model_checkpoint_dir in model_paths:
        test_metrics = eval_chexpert(clip_model_checkpoint_dir, mimic_datadir, verbose=True)
        acc = test_metrics['accuracy']
        f1 = test_metrics['f1']

        table.add_row(model_name, f"{acc:.4f}", f"{f1:.4f}")

    console = Console()
    console.print(table)
