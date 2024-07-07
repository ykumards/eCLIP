# %%
import warnings

warnings.filterwarnings("ignore")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from jinja2 import Template
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
)
import nltk

nltk.download("punkt")

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from eclip.eval.common import load_pl_module
from eclip.transforms.image_transforms import CLIPImageTransform
from eclip.utils.eval_metrics import (
    calculate_bert_score,
    calculate_bleu_score,
    calculate_rouge_scores,
    calculate_sentence_emb_score,
    calculate_meteor_score,
    calculate_chexbert_semb,
)
from eclip.utils import load_config


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


def cluster_sentences(sentences, k, verbose=False):
    _tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    _model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    inputs = _tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        outputs = _model(**inputs)

    # Mean pooling - Take attention mask into account for correct averaging
    input_mask_expanded = (
        inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    )
    sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    sentence_embeddings = mean_embeddings.numpy()
    # Perform KMeans clustering
    num_clusters = k  # Adjust based on your needs
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(
        sentence_embeddings
    )

    # Assign sentences to clusters
    clusters = kmeans.labels_

    # Find representative sentence for each cluster
    representative_sentences = []
    cluster_sentences = [[] for _ in range(num_clusters)]
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_embeddings = sentence_embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]

        # Store sentences of each cluster
        cluster_sentences[cluster_id] = [sentences[i] for i in cluster_indices]

        # Calculate distances of cluster sentences to the cluster center
        distances = cdist([cluster_center], cluster_embeddings, "cosine")[0]

        # Find index of sentence closest to cluster center
        closest_index = cluster_indices[distances.argmin()]
        representative_sentences.append(sentences[closest_index])

    if verbose:
        # Print representative sentences and sentences in each cluster
        for i, (representative, cluster) in enumerate(
            zip(representative_sentences, cluster_sentences)
        ):
            print(f"Cluster {i}:")
            print(f"Representative Sentence: {representative}")
            print("Sentences in the cluster:")
            for sentence in cluster:
                print(f" - {sentence}")
            print("")

    del outputs, inputs
    torch.cuda.empty_cache()
    return representative_sentences


def search_mesh_terms(faiss_index, text_encoder, texts, query, tokenizer, k=1):
    """
    Search for the top k closest MeSH terms to a given query.

    :param index: The FAISS index
    :param text_encoder: The text encoder model (e.g., CLIP's text encoder)
    :param query: The query string
    :param k: Number of closest terms to retrieve
    :return: The top k closest MeSH terms to the query
    """
    text_encoder = text_encoder.to("cuda")
    # Encode the query text
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = inputs.to("cuda")

    with torch.no_grad():
        query_encoder_output = text_encoder(inputs)
        query_embedding = query_encoder_output.projected_vec.cpu().numpy()

    normalized_query_embedding = query_embedding / np.linalg.norm(
        query_embedding, axis=1, keepdims=True
    )

    # Search the index
    D, I = faiss_index.search(normalized_query_embedding, k * 20)

    # Retrieve and print the original findings terms
    closest_terms = [texts[i] for i in I[0]]

    representative_sentences = cluster_sentences(closest_terms, k)

    del inputs
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    return representative_sentences, I


def retrieve_closest_mesh_terms(faiss_index, texts, image_vector, k=1):
    """Retrieve closest MeSH terms for a given image vector"""
    # Ensure the vector is in the right shape (1, dimension)
    if len(image_vector.shape) == 1:
        image_vector = image_vector[None, :]

    # Query the FAISS index
    D, I = faiss_index.search(image_vector, k * 20)

    closest_terms = [texts[i] for i in I[0]]
    representative_sentences = cluster_sentences(closest_terms, k)

    return representative_sentences


# %%
def render_chat_template(tokenizer, messages, bos_token="<s>", eos_token="</s>"):
    template_str = tokenizer.chat_template
    template = Template(template_str)
    return template.render(messages=messages, bos_token=bos_token, eos_token=eos_token)


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, padding_side="left")

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def all_text_metrics(true_reports, generated_reports):
    # BLEU Score
    avg_bleu_score = calculate_bleu_score(true_reports, generated_reports)
    print(f"Average BLEU-2 Score:\t {avg_bleu_score:.3f}")

    avg_meteor_score = calculate_meteor_score(true_reports, generated_reports)
    print(f"Average METEOR Score:\t {avg_meteor_score:.3f}")

    rougue_scores = calculate_rouge_scores(true_reports, generated_reports)
    print(
        f"Average ROUGE Scores:\t ROUGE-1: R={rougue_scores['rouge-1']['r']:.3f}, P={rougue_scores['rouge-1']['p']:.3f}, F={rougue_scores['rouge-1']['f']:.3f}"
    )
    print(
        f"\t\t\t ROUGE-2: R={rougue_scores['rouge-2']['r']:.3f}, P={rougue_scores['rouge-2']['p']:.3f}, F={rougue_scores['rouge-2']['f']:.3f}"
    )
    print(
        f"\t\t\t ROUGE-L: R={rougue_scores['rouge-l']['r']:.3f}, P={rougue_scores['rouge-l']['p']:.3f}, F={rougue_scores['rouge-l']['f']:.3f}"
    )

    bert_P, bert_R, bert_F = calculate_bert_score(true_reports, generated_reports)
    print(f"BERT Score:\t R={bert_R:.3f} P={bert_P:.3f} F={bert_F:.3f}")

    s_emb = calculate_sentence_emb_score(true_reports, generated_reports)
    print(f"SBERT Embedding Similarity:\t {s_emb:.3f}")

    chexbert_s_emb = calculate_chexbert_semb(true_reports, generated_reports)
    print(f"ChexBERT Embedding Similarity:\t {chexbert_s_emb:.3f}")


def clean_text(text):
    # Remove punctuations and numbers
    text = re.sub(r"[^\w\s]|\d", "", text)
    # Strip leading and trailing spaces
    text = text.strip()

    return text

groupnames = ['all', 'normal', 'cardiomegaly', 'opacity', 'atelectasis', 'lung']

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


def main(model_checkpoint_dir, openi_path):
    openi_path = Path(openi_path) if not isinstance(openi_path, Path) else openi_path
    openi_img_path = openi_path / "images/images_normalized"

    df_container = get_openi_dataset(openi_path)

    # train test split over each subgroup
    trains, tests = [], []
    for k, v in df_container.items():
        if k != 'all':
            tr, ts = train_test_split(v, test_size=0.15, random_state=123)
            trains.append(tr)
            tests.append(ts)

    openi_df = pd.concat(trains).reset_index(drop=True)
    openi_df_test = pd.concat(tests).reset_index(drop=True)

    openi_df = openi_df[~openi_df['findings'].isna()]
    openi_df_test = openi_df_test[~openi_df_test['findings'].isna()]

    print(f"train size: {openi_df.shape}")
    print(f"test size: {openi_df_test.shape}")

    cfg = load_config(Path(model_checkpoint_dir) / ".hydra/config.yaml")
    clip_module = load_pl_module(Path(model_checkpoint_dir))
    clip_module.eval()

    # load on cpu to save vram
    device = "cpu"
    clip_module = clip_module.to(device)

    img_transform = CLIPImageTransform(
        cfg.data.img_size, image_mean=cfg.data.mean, image_std=cfg.data.std, is_train=False
    )

    # Initialize the FAISS index
    dimension = cfg.model.proj_dim
    FAISS_INDEX = faiss.IndexFlatIP(dimension)

    unique_findings_reports = openi_df["findings"].apply(lambda row: row.lower().strip()).unique()
    ALL_REPORTS = unique_findings_reports.tolist()

    # New list to hold individual sentences
    ALL_TEXT = set()
    for report in ALL_REPORTS:
        sentences = set([clean_text(report)])
        ALL_TEXT.update(sentences)

    ALL_TEXT = list(ALL_TEXT)
    # Load the tokenizer and model from Hugging Face
    clip_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text.model_name)
    text_encoder = clip_module.text_encoder

    # Process MeSH terms and add to the index in batches
    process_and_add_to_index(FAISS_INDEX, text_encoder, ALL_TEXT, clip_tokenizer)

    # Example usage
    query = "no cardiomegaly"
    closest_mesh_terms, I = search_mesh_terms(
        FAISS_INDEX, text_encoder, ALL_TEXT, query, clip_tokenizer, k=3
    )
    print("Closest MeSH Terms:", closest_mesh_terms)

    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        augmented = img_transform(image=image)
        return augmented

    # model_name = 'meta-llama/Llama-2-7b-chat-hf'
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    llm, llm_tokenizer = load_model(
        model_name=model_name,
    )
    streamer = TextStreamer(llm_tokenizer, skip_prompt=True)

    K = 5  # Number of closest terms to retrieve

    ref_idx = 19
    ref_original_findings1 = openi_df.loc[ref_idx, "findings"]
    ref_processed_image = preprocess_image(
        os.path.join(openi_img_path, openi_df.loc[ref_idx, "filename"])
    )[None, :, :, :]
    ref_image_vector_output = clip_module.image_encoder(ref_processed_image)
    ref_image_vector = ref_image_vector_output.projected_vec.detach().numpy()
    ref_image_vector = ref_image_vector / np.linalg.norm(ref_image_vector, axis=1, keepdims=True)
    ref_closest_mesh_terms = retrieve_closest_mesh_terms(FAISS_INDEX, ALL_TEXT, ref_image_vector, K)
    ref_closest_mesh_string1 = "\n".join(
        [f"• Term {i+1}: {term}" for i, term in enumerate(ref_closest_mesh_terms)]
    )

    ref_idx = 26
    ref_original_findings2 = openi_df.loc[ref_idx, "findings"]
    ref_processed_image = preprocess_image(
        os.path.join(openi_img_path, openi_df.loc[ref_idx, "filename"])
    )[None, :, :, :]
    ref_image_vector_output = clip_module.image_encoder(ref_processed_image)
    ref_image_vector = ref_image_vector_output.projected_vec.detach().numpy()
    ref_image_vector = ref_image_vector / np.linalg.norm(ref_image_vector, axis=1, keepdims=True)
    ref_closest_mesh_terms = retrieve_closest_mesh_terms(FAISS_INDEX, ALL_TEXT, ref_image_vector, K)
    ref_closest_mesh_string2 = "\n".join(
        [f"• Term {i+1}: {term}" for i, term in enumerate(ref_closest_mesh_terms)]
    )

    def llm_report_generate(df, sample_idx):
        processed_image = preprocess_image(
            os.path.join(openi_img_path, df.loc[sample_idx, "filename"])
        )[None, :, :, :]
        image_vector_output = clip_module.image_encoder(processed_image)
        image_vector = image_vector_output.projected_vec.detach().numpy()
        image_vector = image_vector / np.linalg.norm(image_vector, axis=1, keepdims=True)

        closest_mesh_terms = retrieve_closest_mesh_terms(FAISS_INDEX, ALL_TEXT, image_vector, K)
        sample_closest_mesh_terms = "\n".join(
            [f"• Term {i+1}: {term}" for i, term in enumerate(closest_mesh_terms)]
        )
        # sample_closest_mesh_terms = " ".join(f"Closest report {i+1}: {s}\n" for i, s in enumerate(closest_mesh_terms))

        messages = [
            {
                "role": "user",
                "content": "You are to act as a radiologist, trained to generate radiology reports. Your task is to synthesize the information from the closest report snippets provided below into a comprehensive and medically accurate radiologist report for each case. Craft a comprehensive response that is concise, succinct, and focuses on the key findings and potential diagnoses. Your report should maintain a professional tone, with clarity and precision in medical terminology, suitable for medical experts. Remember to be concise, succinct, and focus on the key findings and potential diagnoses, avoiding unnecessary elaboration.",
            },
            {
                "role": "assistant",
                "content": f"Understood. I will use the provided snippets to create accurate and succint radiology reports in a clear and medically polished language.",
            },
            {
                "role": "user",
                "content": f"The following snippets are from reports closely related to the patient's X-ray image. {ref_closest_mesh_string1} Based on these, generate a radiologist report.",
            },
            {"role": "assistant", "content": f"findings: {ref_original_findings1}"},
            {
                "role": "user",
                "content": f"The following snippets are from reports closely related to the patient's X-ray image. {ref_closest_mesh_string2} Based on these, generate a radiologist report.",
            },
            {"role": "assistant", "content": f"findings: {ref_original_findings2}"},
            {
                "role": "user",
                "content": f"The following snippets are from reports closely related to the patient's X-ray image. {sample_closest_mesh_terms} Based on these, generate a radiologist report.",
            },
        ]

        encodeds = llm_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )

        prompt = llm_tokenizer.batch_decode(encodeds)[0]
        model_inputs = encodeds.to(llm.device)

        # generate report!
        generation_config = GenerationConfig(
            max_new_tokens=200,
            num_beams=4,
            do_sample=True,
            top_k=50,
            eos_token_id=llm.config.eos_token_id,
            pad_token_id=llm_tokenizer.eos_token_id,
        )
        generated_ids = llm.generate(
            model_inputs,
            # streamer=streamer, # streamer doesn't work with num_beams > 1
            generation_config=generation_config,
        )
        decoded = llm_tokenizer.batch_decode(generated_ids)[0]

        decoded_split = decoded.split()
        prompt_split = prompt.split()

        generated_report = " ".join(decoded_split[len(prompt_split) :])

        return generated_report

    ## Eval on whole test set
    true_reports, generated_reports = [], []

    report_df = pd.DataFrame(columns=['Problems', 'Generated', 'GT'])

    # openi_df_test = openi_df_test.sample(n=3)
    openi_df_test = openi_df_test.sample(frac=1.0, random_state=123)
    openi_df_test.reset_index(inplace=True, drop=True)

    for idx in tqdm(range(openi_df_test.shape[0])):
        true_report = openi_df_test.loc[idx, "findings"]
        generated_report = llm_report_generate(openi_df_test, idx)

        print(f"Problems: {openi_df_test.loc[idx, 'Problems']}")
        print(true_report)
        print()
        print(generated_report)

        report_df = pd.concat([pd.DataFrame([[openi_df_test.loc[idx, 'Problems'], generated_report, true_report]], columns=report_df.columns), report_df], ignore_index=True)

        true_reports.append(true_report)
        generated_reports.append(generated_report)

    all_text_metrics(true_reports, generated_reports)

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":

    clip_model_checkpoint_dir = "<path>"
    openi_path = "<path>"

    # clip_model_checkpoint_dir = "/projappl/project_462000314/eCLIP/saved_models/eCLIP-best-2024-02-22"
    # openi_path = "/scratch/project_462000314/data_eclip/open-i-raw"

    main(clip_model_checkpoint_dir, openi_path)
