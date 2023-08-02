import numpy as np
import torch
import sys
import os

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional
from tqdm import tqdm

from utils.dataset import Dataset
from utils.model import Model
from utils.processor import Processor
from estimators.estimator import Estimator
from stat_calculators.stat_calculator import StatCalculator, STAT_CALCULATORS, STAT_DEPENDENCIES


def _order_calculators(stats: List[str]) -> List[StatCalculator]:
    ordered: List[StatCalculator] = []
    have_stats: Set[str] = set()
    while len(stats) > 0:
        stat = stats[0]
        if stat in have_stats:
            stats = stats[1:]
            continue
        dependent = False
        if stat not in STAT_DEPENDENCIES.keys():
            raise Exception(f'Cant find stat calculator for: {stat}')
        for d in STAT_DEPENDENCIES[stat]:
            if d not in have_stats:
                stats = [d] + stats
                if stats.count(d) > 40:
                    raise Exception(f'Found possibly cyclic dependencies: {d}')
                dependent = True
        if not dependent:
            stats = stats[1:]
            ordered.append(STAT_CALCULATORS[stat])
            for new_stat in ordered[-1].stats:
                have_stats.add(new_stat)
    return ordered


def _check_unique_names(xs):
    names = set()
    for x in xs:
        if str(x) in names:
            raise Exception(f'Got multiple __str__ values for {x}')
        names.add(str(x))


def _delete_nans(ue, metric):
    new_ue, new_metric = [], []
    for i in range(len(metric)):
        if not np.isnan(metric[i]) and not np.isnan(ue[i]):
            new_ue.append(ue[i])
            new_metric.append(metric[i])
    return new_ue, new_metric


class UEManager:
    def __init__(
            self,
            data: Dataset,
            model: Model,
            estimators: List[Estimator],
            processors: List[Processor],
    ):
        self.model: Model = model
        self.data: Dataset = data
        self.estimators: List[Estimator] = estimators
        _check_unique_names(estimators)
        stats = [s for e in estimators for s in e.stats_dependencies] + \
                ['greedy_tokens', 'greedy_texts']
        self.stat_calculators: List[StatCalculator] = _order_calculators(stats)

        self.gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.stats: Dict[str, List] = defaultdict(list)

        self.processors = processors

    def __call__(self) -> Dict[Tuple[str, str, str, str], float]:
        for inp_texts, target_texts in tqdm(self.data):
            target_tokens = [self.model.tokenizer([text])['input_ids'][0] + [self.model.tokenizer.eos_token_id]
                             for text in target_texts]
            batch_stats: Dict[str, np.ndarray] = {}
            for key, val in [
                ('input_texts', inp_texts),
                ('target_texts', target_texts),
                ('target_tokens', target_tokens),
            ]:
                self.stats[key] += val
                batch_stats[key] = val
            
            batch_stats['generation_params'] = {}
            batch_stats['model'] = self.model
            batch_stats['path_to_train_stats'] = './train_stats.pkl'

            for stat_calculator in self.stat_calculators:
                new_stats = stat_calculator(batch_stats, inp_texts, self.model)
                for stat, stat_value in new_stats.items():
                    if stat in batch_stats.keys():
                        continue
                    batch_stats[stat] = stat_value

            batch_estimations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            for estimator in self.estimators:
                e = estimator(batch_stats).tolist()
                self.estimations[estimator.level, str(estimator)] += e
                batch_estimations[estimator.level, str(estimator)] += e
            batch_gen_metrics: Dict[Tuple[str, str], List[float]] = defaultdict(list)

            for key in ['greedy_texts', 'greedy_tokens']:
                if key in batch_stats.keys():
                    self.stats[key] += batch_stats[key]
            for processor in self.processors:
                processor.on_batch(batch_stats, batch_gen_metrics, batch_estimations)
