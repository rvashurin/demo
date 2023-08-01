from .stat_calculator import StatCalculator, register
from .greedy_probs import GreedyProbsCalculator
from .greedy_lm_probs import GreedyLMProbsCalculator

register(GreedyProbsCalculator())
register(GreedyLMProbsCalculator())
