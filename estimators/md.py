from typing import Union

import torch
from datasets import Dataset as ArrowDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from functools import partial
import os.path
import numpy as np
import pickle

from typing import Dict

from .estimator import Estimator
from utils.md import compute_centroids, \
                     compute_inv_covariance, \
                     mahalanobis_distance_with_known_centroids_sigma_inv

from datasets import load_dataset

def preprocesss_function(example, tokenizer):
    sources = example['email_body'].strip()
    model_inputs = tokenizer(sources, max_length=1024,
                             truncation=True, return_tensors='pt')
    return model_inputs

def _get_dim(model_without_cls_layer):
    return list(model_without_cls_layer.state_dict().items())[-1][1].shape[1]


def fit_md(model):
    tokenizer = model.tokenizer
    model = model.model

    data = load_dataset('aeslc')['train'].select(range(8000))

    device = next(model.parameters()).device
    num_obs = len(data)
    dim = _get_dim(model)

    embeddings = torch.empty((num_obs, dim), dtype=torch.float, device=device)

    possible_input_keys = ["input_ids", "attention_mask"]

    generation_kwargs = {}
    preds = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        start = 0
        for inp in tqdm(data, desc="Embeddings created"):
            sources = inp['email_body'].strip()
            model_inputs = tokenizer(sources, max_length=1024,
                                     truncation=True, return_tensors='pt')
            predictions = model.get_encoder()(
                **model_inputs,
                output_hidden_states=True,
            )
            preds.append(predictions.last_hidden_state.mean(1))
    
    embeddings = torch.cat(preds)

    return embeddings
                

class MD(Estimator):
    def __init__(self):
        super().__init__([], 'sequence')

    def __str__(self):
        return 'MD'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats['model']

        path_to_train_stats = stats['path_to_train_stats']
        
        if os.path.isfile(path_to_train_stats):
            with open(path_to_train_stats, 'rb') as handle:
                centroids, cov = pickle.load(handle)
        else:
            embeddings = fit_md(model)
            
            labels = torch.zeros((embeddings.shape[0],))
            centroids = embeddings.mean(0).unsqueeze(0)
            cov, _ = compute_inv_covariance(centroids, embeddings, labels)
            with open(path_to_train_stats, 'wb') as handle:
                train_stats = pickle.dump([centroids, cov], handle)


        model_inputs = model.tokenizer(stats['input_texts'], max_length=1024,
                                       truncation=True, return_tensors='pt')
        input_embedding = model.model.get_encoder()(
            **model_inputs,
            output_hidden_states=True,
        ).last_hidden_state.mean(1)

        ue = mahalanobis_distance_with_known_centroids_sigma_inv(
            centroids,
            None,
            cov,
            input_embedding
        )

        return ue.detach().numpy()
