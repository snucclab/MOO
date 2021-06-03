from collections import defaultdict
from typing import Iterable, Optional, Dict, Tuple, Iterator, Callable, List, Set

import torch
from numpy import ndarray
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from common.model.const import *
from common.model.types import *
from common.model.util import move_to
from common.solver.const import *
from .ept.attention import MultiheadAttentionWeights
from .ept.chkpt import CheckpointingModule
from .ept.decoder import ExpressionDecoder
from .ept.encoder import TextEncoder
from .ept.util import Squeeze, init_weights, mask_forward, logsoftmax, apply_module_dict

MODEL_CLS = 'model'
OPR_EXCLUDED = {OPR_NEW_EQN_ID}


def _weight_value_metric(params: Iterator[Tuple[str, torch.Tensor]]) -> dict:
    weight_sizes = defaultdict(list)

    for key, weight in params:
        wkey = 'other'
        if 'encoder' in key:
            wkey = 'encoder'
        elif 'decoder' in key:
            wkey = 'decoder'
        elif 'action' in key:
            wkey = 'action'

        if '_embedding' in key:
            wkey += '_embed'
        if 'bias' in key:
            wkey += '_bias'

        weight_sizes[wkey].append(weight.detach().abs().mean().item())

    return {key: sum(values) / len(values) if values else FLOAT_NAN
            for key, values in weight_sizes.items()}


def _length_penalty(score: float, seq_len: int, alpha: float):
    if alpha <= 0:
        return score

    # Following:
    # Wu et al (2016) Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
    penalty = ((5 + seq_len) / (5 + 1)) ** alpha
    return score * penalty


def beam_search(initialize_fn: Callable[[], Tuple[List[dict], torch.Tensor]],
                compute_score_fn: Callable[[int, dict, int], List[Tuple[float, int, dict]]],
                concat_next_fn: Callable[[dict, List[int], dict], dict],
                is_item_finished: Callable[[dict], bool],
                max_len: int = RES_MAX, beam_size: int = 3,
                len_penalty_alpha: float = 0):
    # List of beams. B * [M, ...] and Tensor of [B, M].
    batch, beamscores = initialize_fn()
    finished = False

    # From 1 to HORIZON.
    for seq_len in range(1, max_len):
        if finished:
            break

        next_beamscores = torch.zeros(beamscores.shape[0], beam_size)
        next_batch = []
        finished = True
        for i, item in enumerate(batch):
            if seq_len > 1 and is_item_finished(item):
                # If all beams of this item is done, this item will not be computed anymore.
                next_batch.append(item)
                continue

            # Compute scores
            score_i = [(_length_penalty(score + beamscores[i, m_prev], seq_len, len_penalty_alpha), m_prev, predicted)
                       for score, m_prev, predicted in compute_score_fn(seq_len, item, beam_size)]
            score_i = sorted(score_i, key=lambda t: t[0], reverse=True)[:beam_size]

            # Construct the next beams
            prev_beam_order = [m_prev for _, m_prev, _ in score_i]
            predictions = [prediction for _, _, prediction in score_i]
            next_tokens = {key: torch.LongTensor([prediction[key]
                                                  for prediction in predictions])  # Shape: [M, ?]
                           for key in predictions[0]}

            next_batch.append(concat_next_fn(item, prev_beam_order, next_tokens))
            finished = finished and is_item_finished(next_batch[-1])
            for m_new, (score, m_prev, predicted) in enumerate(score_i):
                next_beamscores[i, m_new] = score

        batch = next_batch
        beamscores = next_beamscores

    return batch


class EPT(CheckpointingModule):
    def __init__(self, **config):
        super().__init__(**config)
        # Encoder: [B, S] -> [B, S, H]
        self.encoder = TextEncoder.create_or_load(encoder=self.config[MDL_ENCODER])

        # Decoder
        self.decoder = ExpressionDecoder.create_or_load(**self.config[MDL_DECODER],
                                                        encoder_config=self.encoder.model.config)

        # Action output
        hidden_dim = self.decoder.hidden_dim
        self.operator = nn.Linear(hidden_dim, OPR_SZ)
        self.operands = nn.ModuleList([nn.ModuleDict({
            '0_attn': MultiheadAttentionWeights(**{MDL_D_HIDDEN: hidden_dim, MDL_D_HEAD: 1}),
            '1_mean': Squeeze(dim=-1)
        }) for _ in range(OPR_MAX_ARITY)])

        # Initialize
        factor = self.decoder.init_factor
        init_weights(self.operator, factor)
        self.operands.apply(lambda w: init_weights(w, factor))

    @property
    def constant_embedding(self):
        # [1, C, H]
        return self.decoder.constant_word_embedding.weight.unsqueeze(0)

    @property
    def operator_embedding(self):
        # [1, V, H]
        return self.decoder.operator_word_embedding.weight.unsqueeze(0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_attention_keys(self, variable: Encoded, number: Encoded,
                              add_constant: bool = True) -> Dict[str, torch.Tensor]:
        # Retrieve size information
        batch_sz, res_len, hidden_dim = variable.vector.shape
        operand_sz = RES_BEGIN + res_len if add_constant else NUM_MAX + res_len

        key = torch.zeros(batch_sz, operand_sz, hidden_dim, device=variable.device)
        key_ignorance_mask = torch.ones(batch_sz, operand_sz, dtype=torch.bool, device=variable.device)
        attention_mask = torch.zeros(res_len, operand_sz, dtype=torch.bool, device=variable.device)

        if add_constant:
            # Assign constant weights
            key[:, :CON_END] = self.constant_embedding
            key_ignorance_mask[:, :CON_END] = False
            offset = CON_END
        else:
            offset = 0

        # Add numbers
        num_count = number.vector.shape[1]
        num_end = offset + num_count
        key[:, offset:num_end] = number.vector
        key_ignorance_mask[:, offset:num_end] = number.pad

        # Add results
        offset = offset + NUM_MAX
        res_end = offset + res_len
        key[:, offset:res_end] = variable.vector
        key_ignorance_mask[:, offset:res_end] = variable.pad

        attention_mask[:, offset:res_end] = mask_forward(res_len, diagonal=0).to(attention_mask.device)
        attention_mask[:, res_end:] = True

        return dict(key=key, key_ignorance_mask=key_ignorance_mask, attention_mask=attention_mask)

    def _encode(self, text: Text) -> Tuple[Encoded, Encoded]:
        return self.encoder(text)

    def _expression_for_train(self, predict_last: bool = False, **kwargs) -> Tuple[tuple, ExpressionPrediction]:
        assert 'text' in kwargs
        assert 'number' in kwargs
        assert 'target' in kwargs
        # text: [B,S]
        # number: [B,N]
        # target: [B,T]
        # decoded: [B,T]
        # ignore the cached result
        decoded, new_cache = self.decoder.forward(**kwargs)

        # Prepare values for attention
        attention_input = self._build_attention_keys(variable=decoded, number=kwargs['number'])

        if predict_last:
            decoded = decoded[:, -1:]
            attention_input['attention_mask'] = attention_input['attention_mask'][-1:]

        # Compute operator
        operator = self.operator(decoded.vector)
        operator[:, :, OPR_NEW_EQN] = NEG_INF
        operator = logsoftmax(operator)

        # Compute operands: List of [B, T, N+T]
        operands = [logsoftmax(apply_module_dict(layer, encoded=decoded.vector, **attention_input))
                    for layer in self.operands]

        return new_cache, ExpressionPrediction.from_tensors(operator, operands)

    def _expression_for_eval(self, **kwargs) -> Expression:
        assert 'text' in kwargs
        assert 'number' in kwargs
        text: Encoded = kwargs['text']
        number: Encoded = kwargs['number']

        def initialize_fn():
            # Initially we start with a single beam.
            batch_sz = text.shape[0]
            beamscores = torch.zeros((batch_sz, 1))

            return [dict(text=text[b:b + 1],  # [1, S]
                         number=number[b:b + 1],  # [1, N]
                         target=Expression.get_generation_base(),  # [1, T=1]
                         cached=None
                         )
                    for b in range(batch_sz)], beamscores

        return self._expression_eval_execute(initialize_fn, **kwargs)

    def _expression_eval_execute(self, initialize_fn, max_len: int = RES_MAX, beam_size: int = 3,
                                 excluded_operators: set = None, **kwargs) -> Expression:
        if excluded_operators is None:
            excluded_operators = set()

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M]
            cached, last_pred = self._expression_for_train(**move_to(beams, self.device), predict_last=True)
            last_pred = last_pred[:, 0].to('cpu')

            # Store cache
            beams['cached'] = move_to(cached, 'cpu')

            # Compute score
            scores = []
            for m_prev in range(last_pred.operator.shape[0]):
                if seq_len > 1 and beams['target'].operator.indices[m_prev, -1].item() in {OPR_DONE_ID, PAD_ID}:
                    scores += [(0, m_prev, dict(operator=[PAD_ID], operands=[PAD_ID] * OPR_MAX_ARITY))]
                    continue

                # Take top-K position for each j-th operand
                operands = [list(zip(*[tensor.tolist()
                                       for tensor in operand_j.log_prob[m_prev].topk(k=k, dim=-1)]))
                            for operand_j in last_pred.operands]

                score_beam = []
                for f, f_info in enumerate(OPR_VALUES):
                    if (f == OPR_DONE_ID and seq_len == 1) or f in excluded_operators:
                        continue

                    arity = f_info[KEY_ARITY]
                    score_f = [(last_pred.operator.log_prob[m_prev, f], (f,))]
                    for operand_j in operands[:arity]:
                        score_f = [(score_aj + score_prev, tuple_prev + (aj,))
                                   for score_aj, aj in operand_j
                                   for score_prev, tuple_prev in score_f]

                    score_beam += [(score, m_prev, dict(operator=[f],
                                                        operands=list(a) + [PAD_ID] * (OPR_MAX_ARITY - len(a))))
                                   for score, (f, *a) in score_f]

                scores += sorted(score_beam, key=lambda t: t[0], reverse=True)[:k]

            return scores

        def concat_next_fn(prev_beams: dict, beam_selected: List[int], list_of_next: dict):
            if prev_beams['target'].shape[0] == 1:
                # Before expanding beams.
                beamsz = len(beam_selected)
                for key in prev_beams:
                    if key in {'target', 'cached'} or prev_beams[key] is None:
                        continue
                    prev_beams[key] = prev_beams[key].repeat(beamsz)

            # Extend beams
            prev_beams['target'] = prev_beams['target'][beam_selected] \
                .extends_to(next_operator=list_of_next['operator'],
                            next_operands=[list_of_next['operands'][:, j:j + 1]
                                           for j in range(OPR_MAX_ARITY)])

            # Select cache of selected beams. All have shape [M, ...], so we will shuffle only the first dim.
            prev_beams['cached'] = tuple(tuple(tensor[beam_selected] for tensor in pair)
                                         for pair in prev_beams['cached'])

            return prev_beams

        def is_all_finished(beams: dict):
            return all(f in {OPR_DONE_ID, PAD_ID}
                       for f in beams['target'].operator.indices[:, -1].tolist())

        with torch.no_grad():
            # Execute beam search. List[Dict[str, ?]]
            batched_beams = beam_search(initialize_fn, compute_next_score_of_beam,
                                        concat_next_fn, is_all_finished, max_len, beam_size)

            # Select top-scored beam
            return Expression.build_batch(*[item['target'][0] for item in batched_beams])

    def forward(self, text: Text, expression: Expression = None, beam: int = 3):
        # Forward the encoder
        text, num_list = self._encode(text.to(self.device))
        num_enc = Encoded.build_batch(*[num.pooled_state for num in num_list])  # [B, N, H]
        return_value = dict()

        if self.training:
            # Compute hidden states & predictions
            return_value['prediction'] = self._expression_for_train(text=text, number=num_enc, target=expression)[-1]
        else:
            # Generate expression
            return_value['expression'] = self._expression_for_eval(text=text, number=num_enc, beam_size=beam)

        return return_value


__all__ = ['EPT']
