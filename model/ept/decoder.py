from typing import Dict, List, Tuple, Optional

import torch
from torch import nn

from common.model.const import *
from common.model.types import Encoded, Expression
from common.solver.const import *
from .attention import *
from .chkpt import *
from .posenc import *
from .util import *


def _adjust_var_index(operand: torch.Tensor, var_begin: int, var_len: List[int]) -> torch.Tensor:
    with torch.no_grad():
        var_bound_tensor = torch.tensor(var_len, dtype=torch.long, device=operand.device).view(-1, 1)
        lower_bound = operand.ge(RES_BEGIN)
        upper_bound = operand.lt(var_bound_tensor + RES_BEGIN)
        bounded_pos = lower_bound.logical_and(upper_bound)

        # Adjust variable operands
        var_diff = RES_BEGIN - var_begin
        operand = operand - (bounded_pos.long() * var_diff)

        # Add variable information to _NEW_VAR()
        # Note: position 0 is _NEW_EQN(), so _NEW_VAR() begin from 1
        var_cond = torch.tensor([[0 < i <= n_var
                                  for i in range(operand.shape[1])]
                                 for n_var in var_len], dtype=torch.bool, device=operand.device)
        var_index = torch.arange(operand.shape[1], dtype=torch.long, device=operand.device) + (var_begin - 1)

        return torch.where(var_cond, var_index, operand)


class ExpressionDecoder(CheckpointingModule):
    """
    Base model for equation generation
    """

    def __init__(self, initialize=True, **config):
        """
        Initiate Equation Builder instance

        :param dict config: Configuration of this model
        """
        super().__init__(**config)

        """ Embedding layers """
        # Look-up table E_f(.) for operator embedding vectors (in Equation 2)
        self.operator_word_embedding = nn.Embedding(OPR_SZ, self.hidden_dim)
        # Positional encoding PE(.) (in Equation 2, 5)
        self.operator_pos_embedding = PositionalEncoding(self.hidden_dim)
        # Vectors representing source: u_num, u_const, u_expr in Equation 3, 4, 5
        self.operand_source_embedding = nn.Embedding(len(SRC_LIST), self.hidden_dim)
        # Look-up table for constants: E_c used in Equation 4
        self.constant_word_embedding = nn.Embedding(CON_END, self.hidden_dim)

        """ Scalar parameters """
        # Initial degrading factor value for c_f and c_a.
        degrade_factor = self.hidden_dim ** 0.5
        # c_f in Equation 2
        self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        # c_a in Equation 3, 4, 5
        self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)

        """ Layer Normalizations """
        # LN_f in Equation 2
        self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        # LN_a in Equation 3, 4, 5
        self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        """ Linear Transformation """
        # Linear transformation from embedding space to hidden space: FF_in in Equation 1.
        self.embed_to_hidden = nn.Linear(self.hidden_dim * (OPR_MAX_ARITY + 1), self.hidden_dim)

        """ Transformer layer """
        # Shared transformer layer for decoding (TransformerDecoder in Figure 2)
        self.shared_decoder_layer = TransformerLayer(hidden_dim=self.hidden_dim,
                                                     intermediate_dim=self.intermediate_dim,
                                                     num_attention_heads=self.num_heads,
                                                     layernorm_eps=self.layernorm_eps)
        # Action output will be defined in other classes

        if initialize:
            """ Initialize weights """
            with torch.no_grad():
                # Initialize Linear, LayerNorm, Embedding
                self.apply(self._init_weights)

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Dimension of hidden vector.
        """
        if MDL_D_HIDDEN in self.config and self.config[MDL_D_HIDDEN] > 0:
            return self.config[MDL_D_HIDDEN]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'hidden_size', DEF_D_HIDDEN)

        return DEF_D_HIDDEN

    @property
    def intermediate_dim(self) -> int:
        """
        :rtype: int
        :return: Dimension of intermediate vector.
        """
        if MDL_D_INTER in self.config and self.config[MDL_D_INTER] > 0:
            return self.config[MDL_D_INTER]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'intermediate_size', DEF_D_INTER)

        return DEF_D_INTER

    @property
    def embedding_dim(self) -> int:
        """
        :rtype: int
        :return: Dimension of embedding vector.
        """
        if MDL_D_EMBED in self.config and self.config[MDL_D_EMBED] > 0:
            return self.config[MDL_D_EMBED]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'embedding_size', DEF_D_EMBED)

        return DEF_D_EMBED

    @property
    def num_hidden_layers(self) -> int:
        """
        :rtype: int
        :return: Number of repetition for applying the same transformer layer
        """
        if MDL_D_LAYER in self.config and self.config[MDL_D_LAYER] > 0:
            return self.config[MDL_D_LAYER]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'num_hidden_layers', DEF_D_LAYER)

        return DEF_D_LAYER

    @property
    def init_factor(self) -> float:
        """
        :rtype: float
        :return: Standard deviation of normal distribution that will be used for initializing weights.
        """
        if MDL_D_INIT in self.config and self.config[MDL_D_INIT] > 0:
            return self.config[MDL_D_INIT]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'initializer_range', DEF_D_INIT)

        return DEF_D_INIT

    @property
    def layernorm_eps(self) -> float:
        """
        :rtype: float
        :return: Epsilon to avoid zero-division in LayerNorm.
        """
        if MDL_D_LN_EPS in self.config and self.config[MDL_D_LN_EPS] > 0:
            return self.config[MDL_D_LN_EPS]
        elif MDL_D_ENC in self.config:
            return getattr(self.config[MDL_D_ENC], 'layer_norm_eps', DEF_D_LN_EPS)

        return DEF_D_LN_EPS

    @property
    def num_heads(self) -> int:
        """
        :rtype: int
        :return: Number of heads in a transformer layer.
        """
        return getattr(self.config[MDL_D_ENC], 'num_attention_heads', DEF_D_HEAD)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights

        :param nn.Module module: Module to be initialized.
        """
        init_weights(module, self.init_factor)

    def _build_operand_embed(self, operand_source: torch.Tensor, operand_value: torch.Tensor,
                             pos_enc: torch.Tensor, number: torch.Tensor) -> torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor operand_source:
            LongTensor containing source information of operands. (This corresponds to a_ij in the paper)
            Shape [B, T, A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor operand_value:
            LongTensor containing content information of operands. (This corresponds to a_ij in the paper)
            Shape [B, T, A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor number:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape [B, N, H], where N = Maximum number of written numbers in the batch.
        :rtype: torch.Tensor
        :return:
            A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
            Shape [B, T, A, H]
        """
        # Compute c_a u_* first.
        operand = get_embedding_without_pad(self.operand_source_embedding, operand_source) * self.operand_source_factor

        # Compute for number operands: [B, T, A, E] (Equation 3)
        number_operand = operand_value.masked_fill(operand_source.ne(SRC_NUMBER), PAD_ID)
        operand += torch.stack([get_embedding_without_pad(number[b], number_operand[b])
                                for b in range(operand_source.shape[0])], dim=0).contiguous()

        # Compute for constant operands: [B, T, A, E] (Equation 4)
        operand += get_embedding_without_pad(self.constant_word_embedding,
                                             operand_value.masked_fill(operand_source.ne(SRC_CONSTANT), PAD_ID))

        # Compute for prior-result operands: [B, T, A, E] (Equation 5)
        prior_result_operand = operand_value.masked_fill(operand_source.ne(SRC_RESULT), PAD_ID)
        operand += get_embedding_without_pad(pos_enc, prior_result_operand)

        return operand

    def _build_decoder_input(self, ids: Expression, number: Encoded, variable: Encoded = None,
                             pos_offsets: Union[List[int], int] = None) -> Encoded:
        """
        Compute input of the decoder, i.e. Equation 1 in the paper.

        :param torch.Tensor ids:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :param torch.Tensor number:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
            Shape: [B, N, H],
            where N = maximum number of written numbers in the batch, and H = dimension of hidden state.
        :rtype: torch.Tensor
        :return: A FloatTensor representing input vector v_i in Equation 1. Shape [B, T, H].
        """
        # Size check
        if len(ids.shape) == 1:
            # Make [T] -> [1, T]
            ids = ids.unsqueeze(0)
        if len(number.shape) == 1:
            # Make [N, H] -> [1, N, H]
            number = number.unsqueeze(0)
        if type(pos_offsets) is int:
            pos_offsets = [pos_offsets]

        operator_ids = ids.operator.indices
        operand_ids = [o.indices for o in ids.operands]
        if variable is not None:
            if len(variable.shape) == 1:
                # Make [V] -> [1, V]
                variable = variable.unsqueeze(0)

            # Treat variable as numbers.
            # - Concat variables right after the numbers
            variable_begin = NUM_BEGIN + number.shape[-1]
            var_lengths = variable.sequence_lengths.tolist()
            number = Encoded.concat(number, variable, dim=1)
            # - Adjust indices
            operand_ids = [_adjust_var_index(o, variable_begin, var_lengths)
                           for o in operand_ids]

        # Operator embedding: [B, T, H] (Equation 2)
        # - compute E_f first
        operator = get_embedding_without_pad(self.operator_word_embedding, operator_ids)

        # - compute PE(.): [T, H]
        len_t = operator_ids.shape[-1]
        if pos_offsets is None:
            position_embeds = self.operator_pos_embedding(len_t)
            operator_pos = position_embeds
        else:
            position_embeds = self.operator_pos_embedding(max(pos_offsets) + len_t)
            position_base = torch.arange(0, len_t, device=position_embeds.device)
            positions = torch.stack([position_base + offset for offset in pos_offsets], dim=0)
            operator_pos = get_embedding_without_pad(position_embeds, positions)
        # - apply c_f and layer norm, and reshape it as [B, T, H]
        operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0))

        # Operand embedding [B, T, A, H] (Equation 3, 4, 5)
        # - prepare source information
        operands = []
        for j, operand_j in enumerate(operand_ids):
            source_id = torch.full_like(operand_j, fill_value=SRC_RESULT) \
                .masked_fill_(operand_j.lt(RES_BEGIN), SRC_NUMBER) \
                .masked_fill_(operand_j.lt(NUM_BEGIN), SRC_CONSTANT) \
                .masked_fill_(operand_j.eq(PAD_ID), PAD_ID)
            operand_id = operand_j \
                         - (source_id.eq(SRC_NUMBER).long() * NUM_BEGIN) \
                         - (source_id.eq(SRC_RESULT).long() * RES_BEGIN)

            # - compute operand embedding [B, T, H]
            operand = self._build_operand_embed(source_id, operand_id, position_embeds, number.pad_fill(0.0))

            # - apply layer norm
            operands.append(self.operand_norm(operand))

        # Concatenate embedding: [B, T, 1+A, H] -> [B, T, (1+A)H]
        operator_operands = torch.stack([operator, *operands], dim=2).contiguous().flatten(start_dim=2)
        # Do linear transformation (Equation 1)
        return Encoded(self.embed_to_hidden(operator_operands), ids.pad)

    def _build_decoder_context(self, embedding: Encoded, text: Encoded = None,
                               prev_key_value: tuple = None) -> Tuple[Encoded, tuple]:
        """
        Compute decoder's hidden state vectors, i.e. d_i in the paper

        :param torch.Tensor embedding:
            FloatTensor containing input vectors v_i. Shape [B, T, H],
            where B = batch size, T = length of decoding sequence, and H = dimension of input embedding
        :param torch.Tensor embedding_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence
            Shape [B, T]
        :param torch.Tensor text:
            FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
            where S = length of input sequence.
        :param torch.Tensor text_pad:
            BoolTensor, whose values are True if corresponding position is PAD in the input sequence
            Shape [B, S]
        :rtype: torch.Tensor
        :return: A FloatTensor of shape [B, T, H], which contains decoder's hidden states.
        """
        # Build forward mask
        mask = mask_forward(embedding.shape[1]).to(embedding.device)
        is_cached = (not self.training) and (prev_key_value is not None)
        # Accumulate KV pairs if this is not on the training phase
        new_key_value = None if self.training else tuple()

        # If not cached, use full embedding [B, T, H] / Otherwise, use only the last embedding [B, 1, H]
        output = embedding.vector if not is_cached else embedding.vector[:, -1:]
        pad = embedding.pad if not is_cached else embedding.pad[:, -1:]

        # Repeatedly pass TransformerDecoder layer
        for _ in range(self.num_hidden_layers):
            if is_cached:
                prev_kv = prev_key_value[:2]
                prev_key_value = prev_key_value[2:]
            else:
                prev_kv = None

            # If cached, output will be [B, 1, H]. Otherwise, [B, T, H].
            output, new_kv = self.shared_decoder_layer(target=output, memory=None if text is None else text.vector,
                                                       target_attention_mask=mask,  # We need full mask
                                                       target_ignorance_mask=embedding.pad,  # We need full mask
                                                       memory_ignorance_mask=None if text is None else text.pad,
                                                       prev_key_value=prev_kv)

            if not self.training:
                # Store new KV pairs into the cache
                new_key_value = new_key_value + new_kv

        # If cached, concatenate with the previous output
        if is_cached:
            assert len(prev_key_value) == 1
            prev_out, prev_pad = prev_key_value[0]
            # [B, T+1, H]
            output = torch.cat([prev_out, output], dim=1).contiguous()
            # [B, T+1]
            pad = torch.cat([prev_pad, pad], dim=1).contiguous()

        # Store output into the cache
        if not self.training:
            new_key_value = new_key_value + ((output, pad),)
            assert len(new_key_value) == self.num_hidden_layers * 2 + 1
            assert all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(t, torch.Tensor) for t in p)
                       for p in new_key_value)

        return Encoded(output, pad), new_key_value

    def forward(self, text: Encoded, number: Encoded, **kwargs) -> Tuple[Encoded, Optional[tuple]]:
        """
        Forward computation of a single beam

        :param dict text:
            Dictionary has followings
            - 'text.encoded': FloatTensor containing encoder's hidden states e_i. Shape [B, S, H],
                where B = batch size, T = length of input sequence, and H = dimension of input embedding.
            - 'text.pad': BoolTensor, whose values are True if corresponding position is PAD in the input sequence
                Shape [B, S]
            - 'number.encoded': FloatTensor containing encoder's hidden states corresponding to numbers in the text.
                (i.e. e_{a_ij} in the paper)
                Shape: [B, N, H], where N = maximum number of written numbers in the batch.
            - 'number.pad': BoolTensor, whose values are True if corresponding position is PAD in the number sequence
                Shape [B, N]
        :param torch.Tensor equation:
            LongTensor containing index-type information of an operator and its operands
            (This corresponds to f_i and a_ij in the paper)
            Shape: [B, T, 1+A], where B = batch size, T = length of expression sequence, and A = maximum arity.
        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of followings
            - 'operator': Log probability of next operators (i.e. Equation 6 without argmax).
                FloatTensor with shape [B, T, F], where F = size of operator vocabulary.
            - '_out': Decoder's hidden states. FloatTensor with shape [B, T, H]
            - '_not_usable': Indicating positions that corresponding output values are not usable in the operands.
                BoolTensor with Shape [B, T].
        """
        # Embedding: [B, T, H]
        target: Expression = kwargs['target']
        # In EPT/FATE, variable will be None. In FESTA, variable will not be None.
        output = self._build_decoder_input(ids=target, number=number, variable=kwargs.get('variable', None))

        # Decoder output: [B, T, H]
        output, new_cached = self._build_decoder_context(embedding=output, text=text,
                                                         prev_key_value=kwargs.get('cached', None))

        # Ignore the result of equality at the function output
        output_not_usable = output.pad.clone()

        previous_op = target.operator.shifted_indices
        output_not_usable[:, :-1].masked_fill_((previous_op >= OPR_EQ_SGN_ID) & (previous_op < OPR_PLUS_ID), True)
        # We need offset '1' because 'function_word' is input and output_not_usable is 1-step shifted output.

        result = Encoded(output.vector, output_not_usable)
        if self.training:
            # We don't need caching
            return result, None
        else:
            return result, new_cached


__all__ = ['ExpressionDecoder']
