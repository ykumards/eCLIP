import numpy as np

import torch
from bert_score import score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer

from eclip.eval.chexbert_labeler import BERT_LABELER

# Ensure you have the necessary NLTK data
nltk.download("punkt")


def calculate_bleu_score(references, candidates):
    """
    Calculate average BLEU-2 score for lists of references and candidates.
    """
    bleu_scores = [
        sentence_bleu([word_tokenize(ref)], word_tokenize(cand), weights=(0.5, 0.5))
        for ref, cand in zip(references, candidates)
    ]
    return np.mean(bleu_scores)


def calculate_rouge_scores(references, candidates):
    """
    Calculate average ROUGE-1, ROUGE-2, and ROUGE-L scores for lists of references and candidates.
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    return scores


def calculate_bert_score(references, candidates, model_type="bert-base-uncased", verbose=False):
    """
    Calculate BERTScore using a pretrained BERT model.
    """
    P, R, F1 = score(candidates, references, model_type=model_type, verbose=verbose)
    return P.mean().item(), R.mean().item(), F1.mean().item()


def calculate_meteor_score(references, candidates):
    """
    Calculate average ROUGE-1, ROUGE-2, and ROUGE-L scores for lists of references and candidates.
    """
    score = meteor_score([references], candidates)
    return score

def calculate_sentence_emb_score(references, candidates):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

    true_embeddings = model.encode(references)
    generated_embeddings = model.encode(candidates)

    true_embeddings = true_embeddings / np.linalg.norm(true_embeddings, axis=1, keepdims=True)
    generated_embeddings = generated_embeddings / np.linalg.norm(generated_embeddings, axis=1, keepdims=True)

    cosine_similarities = np.einsum('ij,ij->i', true_embeddings, generated_embeddings)
    mean_similarity = np.mean(cosine_similarities)

    return mean_similarity


def calculate_chexbert_semb(references, candidates):
    # model_path = "/l/Work/aalto/eCLIP/saved_models/chexbert.pth"
    model_path = "/projappl/project_462000314/eCLIP/saved_models/chexbert.pth"
    labeler = BERT_LABELER()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    checkpoint = torch.load(model_path)
    bert_weights = {}
    for key in checkpoint['model_state_dict']:
        if key.startswith('module.bert.'):
            bert_weights[key[12:]] = checkpoint['model_state_dict'][key]

    labeler.bert.load_state_dict(bert_weights)
    # Function to compute embeddings
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs, cls_hideen = labeler(inputs['input_ids'], inputs['attention_mask'])
        return cls_hideen.detach().numpy()  # Using the [CLS] token representation

    # Compute embeddings for references and candidates
    reference_embeddings = np.vstack([get_embedding(ref) for ref in references])
    candidate_embeddings = np.vstack([get_embedding(cand) for cand in candidates])

    # Normalize embeddings
    reference_embeddings /= np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

    cosine_similarities = np.einsum('ij,ij->i', reference_embeddings, candidate_embeddings)
    mean_similarity = np.mean(cosine_similarities)

    return mean_similarity


if __name__ == "__main__":
    # Example usage
    references = [
        "This is a test sentence for evaluation.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    candidates = ["This is a test line for evaluation.", "A fast brown fox leaps over a lazy dog."]

    # BLEU Score
    avg_bleu_score = calculate_bleu_score(references, candidates)
    print(f"Average BLEU-2 Score:\t {avg_bleu_score:.3f}")

    rougue_scores = calculate_rouge_scores(references, candidates)
    print(
        f"Average ROUGE Scores:\t ROUGE-1: R={rougue_scores['rouge-1']['r']:.3f}, P={rougue_scores['rouge-1']['p']:.3f}, F={rougue_scores['rouge-1']['f']:.3f}"
    )
    print(
        f"\t\t\t ROUGE-2: R={rougue_scores['rouge-2']['r']:.3f}, P={rougue_scores['rouge-2']['p']:.3f}, F={rougue_scores['rouge-2']['f']:.3f}"
    )
    print(
        f"\t\t\t ROUGE-L: R={rougue_scores['rouge-l']['r']:.3f}, P={rougue_scores['rouge-l']['p']:.3f}, F={rougue_scores['rouge-l']['f']:.3f}"
    )

    bert_P, bert_R, bert_F = calculate_bert_score(references, candidates)
    print(f"BERT Score:\t R={bert_R:.3f} P={bert_P:.3f} F={bert_F:.3f}")
