import os
from datetime import datetime
import psutil
import GPUtil
import torch
import torch.nn as nn
import socket
import random
import re
import numpy as np
from omegaconf import OmegaConf


def manual_seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["WDS_SEED"] = str(seed)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def gethostname():
    return socket.gethostname()


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    return device


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def configure_paths_hg(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_save_dir_root = config.output_save_dir_root
    model_name = config.model_name

    config.output_dir = f"{output_save_dir_root}/{model_name}/at_{timestamp}/output"
    config.log_dir = f"{output_save_dir_root}/{model_name}/at_{timestamp}/log"
    config.save_model_path = f"{output_save_dir_root}/{model_name}/at_{timestamp}/"

    # Ensure the directories exist
    ensure_dir_exists(config.output_dir)
    ensure_dir_exists(config.log_dir)
    ensure_dir_exists(config.save_model_path)

    return config


def configure_paths(config):
    # This path points to the dynamically created directory by Hydra
    output_save_dir_root = os.getcwd()
    # Note: wandb will log to cwd

    config.paths.output_dir = output_save_dir_root
    config.paths.save_model_path = f"{output_save_dir_root}/saved_model/"

    # Ensure the directories exist
    ensure_dir_exists(config.paths.output_dir)
    ensure_dir_exists(config.paths.save_model_path)

    return config


def load_config(path_to_config):
    return OmegaConf.load(path_to_config)


def unnormalize(tensor, mean, std):
    """
    Unnormalize a given tensor.

    Parameters:
    - tensor (torch.Tensor): The tensor to unnormalize.
    - mean (list or tuple): The mean used to normalize the tensor.
    - std (list or tuple): The standard deviation used to normalize the tensor.

    Returns:
    torch.Tensor: The unnormalized tensor.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def track_ram_usage():
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = GPUtil.getGPUs()[0].memoryUtil if GPUtil.getGPUs() else 0  # GPU usage in percentage
    return ram_usage, gpu_usage


def extract_sentences_from_full_text(transcript_text):
    # This regular expression captures sentences ending with ., !, or ?
    sentence_pattern = re.compile(r"[^.!?]*[.!?]")
    return [match.group(0).strip() for match in sentence_pattern.finditer(transcript_text)]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True