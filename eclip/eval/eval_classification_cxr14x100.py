import warnings

warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from rich.console import Console
from rich.table import Table


import eclip.constants as const
from eclip.eval.common import load_pl_module
from eclip.utils import load_config, manual_seed_everything
from eclip.transforms.image_transforms import CLIPImageTransform
from eclip.data import ChestXRay14x100Dataset


def get_cxr14_prompts_ensemble(tokenizer):
    prompt_keys = {
        0: "an x-ray of a lung with",
        1: "Chest X-ray depicting signs of",
        2: "Radiology image of a lung affected by",
    }
    prompt_ens = {}
    for k, v in prompt_keys:
        prompts = [f"{v} {label_name.lower()}" for label_name in const.CXR14x100_LABELS]
        tokenized_prompts = tokenizer(
            prompts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ens[k] = tokenized_prompts
    return prompt_ens


def get_cxr14_prompts(tokenizer):
    prompts = [
        f"an x-ray of a lung with {label_name.lower()}" for label_name in const.CXR14x100_LABELS
    ]
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return tokenized_prompts


def evaluate_testset(model, text_embeddings, test_loader, class_labels, device, verbose=False):
    text_embeddings = text_embeddings.to(device)
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    iterable = (
        tqdm(test_loader, desc="Evaluating on CXR-14", leave=False) if verbose else test_loader
    )
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

            # Convert similarity scores to probabilities
            predicted_labels = np.argmax(similarity_scores.cpu().numpy(), axis=1)

            # all_preds.append(probabilities.cpu().numpy())
            all_preds.append(predicted_labels)
            all_labels.append(labels.numpy())

    # Concatenate all batch results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_labels, all_preds


def get_cxr_metrics(all_labels, all_preds):
    converted_preds = np.eye(all_labels.shape[1])[all_preds]

    accuracy = accuracy_score(all_labels, converted_preds)
    f1 = f1_score(all_labels, converted_preds, average="macro")

    return {"accuracy": accuracy, "f1": f1}


def eval_cxr14(path_to_training_output_folder, cxr14_path, verbose=False):
    path_to_training_output_folder = (
        Path(path_to_training_output_folder)
        if not isinstance(path_to_training_output_folder, Path)
        else Path
    )

    cfg = load_config(path_to_training_output_folder / ".hydra/config.yaml")

    clip_module = load_pl_module(
        path_to_training_output_folder, use_expert=cfg.use_expert
    )
    clip_module.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    clip_module = clip_module.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_name)
    img_transform = CLIPImageTransform(
        cfg.data.img_size, image_mean=cfg.data.mean, image_std=cfg.data.std, is_train=False
    )

    test_ds = pd.read_pickle(cxr14_path)

    test_dataset = ChestXRay14x100Dataset(
        test_ds,
        transform=img_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=4
    )
    cxr14_prompts = get_cxr14_prompts(tokenizer=tokenizer)
    cxr14_prompts = {k: v.to(device) for k, v in cxr14_prompts.items()}

    with torch.no_grad():
        text_encoder_output = clip_module.text_encoder(cxr14_prompts)
        text_embeddings = text_encoder_output.projected_vec
        normed_text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    all_labels, all_preds = evaluate_testset(
        clip_module,
        normed_text_embeddings,
        test_loader,
        const.CXR14x100_LABELS,
        device="cuda",
        verbose=verbose,
    )
    test_metrics = get_cxr_metrics(all_labels, all_preds)

    return test_metrics


if __name__ == "__main__":
    manual_seed_everything(2024)

    cxr14_path = ""

    model_paths = [
        ("CLIP", ""),
        ("eCLIP", "")

    ]

    table = Table(title="Zero-shot Classification \n (CXR 14x100)")
    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Acc", justify="right", style="magenta")
    table.add_column("F1 Score", justify="right", style="magenta")

    for model_name, clip_model_checkpoint_dir in model_paths:
        test_metrics = eval_cxr14(clip_model_checkpoint_dir, cxr14_path, verbose=True)
        acc = test_metrics['accuracy']
        f1 = test_metrics['f1']

        table.add_row(model_name, f"{acc:.4f}", f"{f1:.4f}")

    console = Console()
    console.print(table)
