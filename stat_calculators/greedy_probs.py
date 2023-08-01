import torch
import numpy as np

from typing import Dict, List

from stat_calculators.stat_calculator import StatCalculator
from utils.model import Model


class GreedyProbsCalculator(StatCalculator):
    def __init__(self):
        super().__init__(['input_texts', 'input_tokens',
                          'greedy_log_probs', 'greedy_tokens',
                          'greedy_texts', 'attention', 'greedy_log_likelihoods'], [])

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model) -> Dict[str, np.ndarray]:
        inp_tokens = model.tokenizer(texts)
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=256,
                min_length=2,
                output_attentions=True,
            )
            logits = torch.stack(out.scores, dim=1).log_softmax(-1)
            if (out.__class__.__name__ == 'GreedySearchEncoderDecoderOutput') or \
               (out.__class__.__name__ == 'BeamSearchEncoderDecoderOutput'):
                attentions = out.decoder_attentions
            else:
                attentions = out.attentions
            sequences = out.sequences

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        for i in range(len(texts)):
            # For CausalLM models
            #seq = sequences[i, batch['input_ids'].shape[1]:].cpu()
            # For BART
            seq = sequences[i, 1:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_texts.append(model.tokenizer.decode(seq[:text_length], skip_special_tokens=True))

        attn_mask = []
        for i in range(len(texts)):
            c = len(cut_sequences[i])
            attn_mask.append(np.zeros(shape=(c, c)))
            for j in range(1, c):
                attn_mask[i][j, :j] = torch.vstack(
                    [attentions[j][l][i][h][0][-j:]
                     for l in range(len(attentions[j]))
                     for h in range(len(attentions[j][l][i]))]).mean(0).cpu().numpy()

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        return {
            'input_texts': texts,
            'input_tokens': inp_tokens,
            'greedy_log_probs': cut_logits,
            'greedy_tokens': cut_sequences,
            'greedy_texts': cut_texts,
            'attention': attn_mask,
            'greedy_log_likelihoods': ll,
        }
