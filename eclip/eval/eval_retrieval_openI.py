import os
from pathlib import Path

import re
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image


import eclip.constants as const
from eclip.eval.common import load_pl_module
from eclip.utils import load_config
from eclip.transforms.image_transforms import CLIPImageTransform

torch.set_float32_matmul_precision("medium")


def get_openi_dataset(openi_path):
    openi_df = pd.read_csv(openi_path / "indiana_reports.csv")
    openi_projections = pd.read_csv(openi_path / "indiana_projections.csv")
    openi_projections = openi_projections.query("projection == 'Frontal'")
    openi_df = openi_df.merge(openi_projections, on='uid', how='inner')

    openi_df['processed_problems'] = openi_df['Problems'].apply(lambda row: row.split(";"))
    openi_df['n_problems'] = openi_df['processed_problems'].apply(lambda row: len(row))

    openi_df['cardiomegaly'] = openi_df['Problems'].apply(lambda row: int('cardiomegaly' in row.lower()))
    openi_df['lung'] = openi_df['Problems'].apply(lambda row: int('lung' in row.lower()))
    openi_df['opacity'] = openi_df['Problems'].apply(lambda row: int('opacity' in row.lower()))
    openi_df['atelectasis'] = openi_df['Problems'].apply(lambda row: int('atelectasis' in row.lower()))

    normal_df = openi_df[openi_df['Problems'] == 'normal']
    cardiomegaly_df = openi_df[openi_df['cardiomegaly'] == 1]
    lung_df = openi_df[openi_df['lung'] == 1]
    opacity_df = openi_df[openi_df['opacity'] == 1]
    atelectasis_df = openi_df[openi_df['atelectasis'] == 1]

    df_container = {
        'all': openi_df,
        'normal': normal_df,
        'cardiomegaly': cardiomegaly_df,
        'opacity': opacity_df,
        'atelectasis': atelectasis_df,
        'lung': lung_df
    }

    return df_container

def clean_text(text):
    # Remove punctuations and numbers
    text = re.sub(r"[^\w\s]|\d", "", text)
    # Strip leading and trailing spaces
    text = text.strip()

    return text


def process_and_add_to_index(faiss_index, text_encoder, texts, tokenizer, batch_size=128):
    """Process texts in batches and add embeddings to the FAISS index"""
    text_encoder = text_encoder.to("cuda")
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing and adding batches"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        inputs = inputs.to("cuda")
        with torch.no_grad():
            text_encoder_output = text_encoder(inputs)
            text_embeddings = text_encoder_output.projected_vec.cpu().numpy()
        norm_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        faiss_index.add(norm_embeddings)  # Add normalized embeddings to the index
    del inputs, norm_embeddings
    torch.cuda.empty_cache()
    return faiss_index


def retrieve_closest_mesh_terms(faiss_index, texts, image_vector, k=1):
    """Retrieve closest MeSH terms for a given image vector"""
    # Ensure the vector is in the right shape (1, dimension)
    if len(image_vector.shape) == 1:
        image_vector = image_vector[None, :]

    # Query the FAISS index
    D, I = faiss_index.search(image_vector, k)

    closest_terms = [texts[i] for i in I[0]]
    return closest_terms


def retrieve_best_report(path_to_output_folder, openi_path):
    openi_path = Path(openi_path) if not isinstance(openi_path, Path) else openi_path
    openi_img_path = openi_path / "images/images_normalized"

    df_container = get_openi_dataset(openi_path)
    openi_df = df_container['all']
    openi_df = openi_df[~openi_df['findings'].isna()]

    clip_config = load_config(Path(path_to_output_folder) / ".hydra/config.yaml")

    clip_module = load_pl_module(
        path_to_output_folder, use_expert=clip_config.use_expert
    )
    clip_module.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    clip_module = clip_module.to(device)

    img_transform = CLIPImageTransform(
        clip_config.data.img_size,
        image_mean=clip_config.data.mean,
        image_std=clip_config.data.std,
        is_train=True,
    )

    dimension = clip_config.model.proj_dim
    FAISS_INDEX = faiss.IndexFlatIP(dimension) # Becomes cosine sim if vectors are normed

    unique_findings_reports = openi_df["findings"].apply(lambda row: row.lower().strip()).unique()
    ALL_TEXT = unique_findings_reports.tolist()

    clip_tokenizer = AutoTokenizer.from_pretrained(clip_config.model.text.model_name)
    text_encoder = clip_module.text_encoder

    FAISS_INDEX = process_and_add_to_index(FAISS_INDEX, text_encoder, ALL_TEXT, clip_tokenizer)

    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        augmented = img_transform(image=image)
        return augmented

    # we loop through the dataframe and for each image, retrieve the closest findings reports
    # pick top 1, 5 and 10
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total_queries = 0
    for idx, row in tqdm(openi_df.iterrows(), total=openi_df.shape[0]):
        image_path = row['filename']
        image = preprocess_image(
            os.path.join(openi_img_path, image_path)
        )[None, :, :, :]
        image = image.to(clip_module.device)
        image_vector_output = clip_module.image_encoder(image)
        image_vector = image_vector_output.projected_vec.detach().cpu().numpy()
        image_vector = image_vector / np.linalg.norm(image_vector, axis=1, keepdims=True)

        true_report = row['findings'].lower().strip()

        retrieved_reports  = retrieve_closest_mesh_terms(FAISS_INDEX, ALL_TEXT, image_vector, k=10)
        if true_report in retrieved_reports[:1]:
            recall_at_1 += 1
        if true_report in retrieved_reports[:5]:
            recall_at_5 += 1
        if true_report in retrieved_reports[:10]:
            recall_at_10 += 1

        total_queries += 1

    recall_at_1 = recall_at_1 / total_queries
    recall_at_5 = recall_at_5 / total_queries
    recall_at_10 = recall_at_10 / total_queries

    print(f"Recall@1: {recall_at_1}")
    print(f"Recall@5: {recall_at_5}")
    print(f"Recall@10: {recall_at_10}")


if __name__ == "__main__":
    path_to_output_folder = "<path>"
    data_path = "<path>"

    retrieve_best_report(path_to_output_folder, openi_path=data_path)
