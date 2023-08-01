import json
import sys

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from estimators import *
from utils.model import Model


def parse_ue_method(method_name: str, model_path: str, cache_path: str) -> Estimator:
    match method_name:
        case "sequence-level, md":
            return MD()
        case _:
            raise Exception(f'Unknown method: {method_name}')


def parse_model(model: str) -> str:
    match model:
        case 'BART-aeslc':
            return 'rvashurin/bart_base_aeslc'
        case _:
            raise Exception(f'Unknown model: {model}')

