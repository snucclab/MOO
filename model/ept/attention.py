from typing import Tuple

import torch
from torch import nn

from common.model.const import *


class MultiheadAttentionWeights(nn.Module):
    """
    Class for computing multi-head attention weights (follows the paper, 'Attention is all you need')

    This class computes dot-product between query Q and key K, i.e.

    .. math::
        \\frac{Q^\\top K}{\\sqrt{D}}
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttentionWeights class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default.
        :keyword int num_heads: Number of attention heads (N). 12 by default.
        """
        super().__init__()
        self.config = config

        # Check whether D is divisible by H.
        assert self.hidden_dim % self.num_heads == 0, \
            "Hidden dimension %s is not divisible by the number of heads %s." % (self.hidden_dim, self.num_heads)

        # Linear transform for query Q
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Linear transform for key K
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Vector dimension D of input of a single attention head
        self.dim_head = self.hidden_dim // self.num_heads
        # Square root of vector dimension, i.e. \\sqrt{D}
        self.sqrt_dim = self.dim_head ** 0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, prev_key: torch.Tensor = None, head_at_last: bool = True,
                is_self: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention weights

        :param torch.Tensor query:
            FloatTensor representing the query matrix Q with shape [B, S, H],
            where B = batch size, S = query sequence length, and H = vector dimension of hidden states.
        :param torch.Tensor key:
            FloatTensor representing the key matrix K with shape [B, T, H] or [1, T, H], where T = key sequence length
            By default, this is `None` (Use query matrix Q as a key matrix)
        :param torch.Tensor key_ignorance_mask:
            BoolTensor representing the mask for ignoring column vector in matrix K, with shape [B, T].
            If an element at (b, t) is `True,` then all return elements at B=b, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param torch.Tensor attention_mask:
            BoolTensor representing Attention mask for ignoring a key for each query item, with shape [S, T].
            If an element at (s, t) is `True,` then all return elements at S=s, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param bool head_at_last:
            Use `True` to make shape of return value be [B, S, T, N], where N = number of attention heads.
            If `False,` this method will return [B, N, S, T].
            By default, this is `True`
        :rtype: torch.FloatTensor
        :return: FloatTensor of Multi-head Attention weights
        """
        # Query: [B, S/1, H], Key: [B, T/1, H]
        query, key = self._prepare_qk(query, key)

        if prev_key is not None:
            # We have cached results.
            if not is_self:
                # Cross-attention. Key doesn't change.
                key = prev_key
            else:
                # Self-attention. Key should be extended.
                # [B, T, H] + [B, 1, H] => [B, T+1, H]
                key = torch.cat([prev_key, key], dim=1).contiguous()

        # Verify the shape of masks
        assert key_ignorance_mask is None or (key.shape[:2] == key_ignorance_mask.shape and
                                              key_ignorance_mask.dtype == torch.bool)
        assert attention_mask is None or (query.shape[1] == attention_mask.shape[0] and
                                          key.shape[1] == attention_mask.shape[1] and
                                          attention_mask.dtype == torch.bool)

        # Return as tuple
        return self._compute_attention_weight(query, key, key_ignorance_mask, attention_mask, head_at_last), key

    def _compute_attention_weight(self, query, key, key_ignorance_mask, attention_mask, head_at_last):
        batch_size, query_len = query.shape[:2]
        key_len = key.shape[1]

        # Transform query [B, S, N, H/N] -> [B, N, S, H/N] -> [BN, S, H/N].
        query = query.view(batch_size, query_len, self.num_heads, self.dim_head) \
            .transpose(1, 2).flatten(0, 1).contiguous()

        # Transform key [B, T, N, H/N] -> [B, N, H/N, T] -> [BN, H/T, T].
        key = key.view(batch_size, key_len, self.num_heads, self.dim_head) \
            .permute(0, 2, 3, 1).flatten(0, 1).contiguous()

        # Compute attention weights: [BN, S, T] -> [B, N, S, T]
        attention_weights = torch.bmm(query, key).view(batch_size, self.num_heads, query_len, key_len).contiguous()

        # Apply masks (IMPORTANT!!! This should be applied after GELU for output weights)
        if attention_mask is not None:
            # Recap: attention mask has shape [S, T], which can be broadcasted as [1, 1, S, T].
            attention_weights.masked_fill_(attention_mask, NEG_INF)

        if key_ignorance_mask is not None:
            # Recap: ignorance mask has shape [B, T] -> [B, 1, 1, T] and apply it.
            attention_weights.masked_fill_(key_ignorance_mask.unsqueeze(1).unsqueeze(1), NEG_INF)

        if head_at_last:
            # Output will be [B, N, S, T] -> [B, S, T, N]
            return attention_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attention_weights

    def _prepare_qk(self, query, key):
        # If key is None, reuse query matrix Q.
        if key is None:
            key = query

        # Check size & type conditions
        assert query.shape[0] == key.shape[0] or key.shape[0] == 1 or query.shape[0] == 1
        # Store length information
        batch_size = max(key.shape[0], query.shape[0])

        # Project query & key with linear transformations
        query = self.linear_q(query)
        key = self.linear_k(key)

        # Scale query with sqrt(dim)
        query = query / self.sqrt_dim

        # If key / value has shape [1, T, H], expand it.
        if query.shape[0] == 1:
            query = query.expand(batch_size, -1, -1)
        if key.shape[0] == 1:
            key = key.expand(batch_size, -1, -1)

        return query, key

    @property
    def hidden_dim(self) -> int:
        """
        :rtype: int
        :return: Vector dimension of hidden states (H)
        """
        return self.config.get(MDL_D_HIDDEN, DEF_D_HIDDEN)

    @property
    def num_heads(self) -> int:
        """
        :rtype: int
        :return: Number of attention heads (N)
        """
        return self.config.get(MDL_D_HEAD, DEF_D_HEAD)


class MultiheadAttention(nn.Module):
    """
    Class for computing multi-head attention (follows the paper, 'Attention is all you need')

    This class computes attention over K-V pairs with query Q, i.e.

    .. math::
        \\textrm{softmax}\\left(\\frac{Q^\\top K}{\\sqrt{D}}\\right) V
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttention class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default
        :keyword int num_heads: Number of attention heads (N). 12 by default
        :keyword float dropout_p: Probability of dropout. 0 by default
        """
        super().__init__()
        # Multi-head Attention Weight layer
        self.attn = MultiheadAttentionWeights(**config)
        # Linear transformations for value and output matrix.
        self.linear_v = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)
        self.linear_out = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, key_ignorance_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None, prev_key_value: tuple = None, is_self: bool = False):
        """
        Compute multi-head attention

        :param torch.Tensor query:
            FloatTensor representing the query matrix Q with shape [B, S, H],
            where B = batch size, S = query sequence length, and H = vector dimension of hidden states.
        :param torch.Tensor key_value:
            FloatTensor representing the key matrix K or value matrix V with shape [B, T, H] or [1, T, H],
            where T = key sequence length.
            By default, this is `None` (Use query matrix Q as a key matrix)
        :param torch.Tensor key_ignorance_mask:
            BoolTensor representing the mask for ignoring column vector in matrix K, with shape [B, T].
            If an element at (b, t) is `True,` then all return elements at B=b, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :param torch.Tensor attention_mask:
            BoolTensor representing Attention mask for ignoring a key for each query item, with shape [S, T].
            If an element at (s, t) is `True,` then all return elements at S=s, T=t will set to be -Infinity.
            By default, this is `None` (There's no mask to apply)
        :rtype: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]
        :return:
            If head_at_last is True, return (Attention Output, Attention Weights).
            Otherwise, return only the Attention Output
            - Attention Output: Shape [B, S, H].
            - Attention Weights: Shape [B, S, T, N].
        """
        # If key_value is None, reuse query matrix Q.
        if key_value is None:
            key_value = query

        if prev_key_value is not None:
            # Case: Cached.
            attn_weights, new_key = self.attn(query=query, key_ignorance_mask=key_ignorance_mask,
                                              attention_mask=attention_mask, prev_key=prev_key_value[0],
                                              head_at_last=False, is_self=is_self)
        else:
            # Case: Not-cached.
            # Compute attention scores: [B, N, S, T].
            attn_weights, new_key = self.attn(query=query, key=key_value, key_ignorance_mask=key_ignorance_mask,
                                              attention_mask=attention_mask, head_at_last=False)

        # Retrieve shape [B, N, S, T]
        batch_size, _, query_len, key_len = attn_weights.shape

        # Compute Softmax values. Shape [B, N, S, T] -> [BN, S, T].
        # For numerical stability, replace NaN with -Inf. (NaN occurs when we should ignore all weights.)
        attn = attn_weights.softmax(dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0).flatten(0, 1)

        if prev_key_value is not None:
            if not is_self:
                # Cross-attention. Value doesn't change [B, N, T, H/N]
                new_value = prev_key_value[1]
            else:
                # Self-attention, Value will be extended [B, N, T+1, H/N]
                prev_value = prev_key_value[1]
                new_value = torch.cat([prev_value, self._transform_value(key_value, batch_size)],
                                      dim=2).contiguous()
        else:
            # Not cached. Just compute it.
            new_value = self._transform_value(key_value, batch_size)

        # Flatten dim #0 and #1: [B, N, T, H/N] -> [BN, T, H/N].
        value = new_value.flatten(0, 1).contiguous()

        # Compute output of weighted sum: [BN, S, H/N] -> [B, N, S, H/N] -> [B, S, N, H/N] -> [B, S, H].
        output = torch.bmm(attn, value) \
            .view(batch_size, self.attn.num_heads, query_len, self.attn.dim_head) \
            .transpose(1, 2).flatten(2, 3).contiguous()

        # Map outputs and return. [B, S, H].
        output = self.linear_out(output)
        return output, (new_key, new_value)

    def _transform_value(self, value, batch_size):
        # Retrieve shape
        value_batch, key_len = value.shape[:2]

        # Pass linear and transpose value matrix: [1 or B, T, N, H/N] -> [1 or B, N, T, H/N].
        value = self.linear_v(value) \
            .view(value_batch, key_len, self.attn.num_heads, self.attn.dim_head).transpose(1, 2)

        # If value has shape [1, *], expand it.
        if value_batch == 1:
            value = value.expand(batch_size, -1, -1, -1)

        # [B, N, T, H/N]
        return value


class TransformerLayer(nn.Module):
    """
    Class for Transformer Encoder/Decoder layer (follows the paper, 'Attention is all you need')
    """

    def __init__(self, hidden_dim: int = 768, intermediate_dim: int = 2048,
                 num_attention_heads: int = 8, layernorm_eps: float = 1E-12, cross_only: bool = False):
        """
        Initialize TransformerLayer class

        :param ModelConfig config: Configuration of this Encoder/Decoder layer
        """
        super().__init__()
        # Flag for cross-attention only layer
        self.cross_only = cross_only

        if not self.cross_only:
            # Self-attention layer
            self.attn = MultiheadAttention(**{MDL_D_HIDDEN: hidden_dim, MDL_D_HEAD: num_attention_heads,
                                              MDL_D_LN_EPS: layernorm_eps})
        # Source-Target attention layer
        self.mem = MultiheadAttention(**{MDL_D_HIDDEN: hidden_dim, MDL_D_HEAD: num_attention_heads,
                                         MDL_D_LN_EPS: layernorm_eps})

        # Linear transformation layer for expansion (H -> I) where I = vector dimension of intermediate state
        self.lin_expand = nn.Linear(hidden_dim, intermediate_dim)
        # Linear transformation layer for output (I -> H)
        self.lin_collapse = nn.Linear(intermediate_dim, hidden_dim)
        # GELU layer
        self.gelu = nn.GELU()

        if not self.cross_only:
            # Post Layer Normalization for self-attention
            self.norm_attn = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        # Post Layer Normalization for source-target attention
        self.norm_mem = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        # Post Layer Normalization for outputting
        self.norm_out = nn.LayerNorm(hidden_dim, eps=layernorm_eps)

    def forward(self, target, target_ignorance_mask=None, target_attention_mask=None,
                memory=None, memory_ignorance_mask=None, memory_attention_mask=None, prev_key_value=None):
        """
        Forward-computation of Transformer Encoder/Decoder layers

        :param torch.Tensor target:
            FloatTensor indicating Sequence of target vectors. Shape [B, T, H]
            where B = batch size, T = length of target sequence, H = vector dimension of hidden state
        :param torch.Tensor target_ignorance_mask:
            BoolTensor indicating Mask for target tokens that should be ignored. Shape [B, T].
        :param torch.Tensor target_attention_mask:
            BoolTensor indicating Target-to-target Attention mask for target tokens. Shape [T, T].
        :param torch.Tensor memory:
            FloatTensor indicating Sequence of source vectors. Shape [B, S, H]
            where S = length of source sequence
            This can be None when you want to use this layer as an encoder layer.
        :param torch.Tensor memory_ignorance_mask:
            BoolTensor indicating Mask for source tokens that should be ignored. Shape [B, S].
        :rtype: torch.FloatTensor
        :return: Decoder hidden states per each target token, shape [B, S, H].
        """
        # Compute self-attention
        new_kv_tuples = tuple()
        is_cached = (not self.training) and (prev_key_value is not None)
        if not self.cross_only:
            if is_cached:
                # Self attention: K/V will be expanded.
                # Query should be the last entry of the key [B, 1, H]
                # Key-ignorance mask should be full-sized.
                # Attention mask should be [1, T].
                attented, new_kv = self.attn(query=target[:, -1:], attention_mask=target_attention_mask[-1:],
                                             key_ignorance_mask=target_ignorance_mask,
                                             prev_key_value=prev_key_value[0], is_self=True)
            else:
                attented, new_kv = self.attn(query=target, attention_mask=target_attention_mask,
                                             key_ignorance_mask=target_ignorance_mask)
            target = target + attented
            target = self.norm_attn(target)

            # Store other return values
            new_kv_tuples = new_kv_tuples + (new_kv,)
        else:
            assert memory is not None

        # Compute attention over targets with source as queries.
        if memory is not None:
            if is_cached:
                # Cached cross-attention: K/V is fixed.
                # Query should be the last entry of the key [B, 1, H]
                attented, new_kv = self.mem(query=target[:, -1:], prev_key_value=prev_key_value[-1])
            else:
                attented, new_kv = self.mem(query=target, attention_mask=memory_attention_mask,
                                            key_value=memory, key_ignorance_mask=memory_ignorance_mask)
            target = target + attented
            target = self.norm_mem(target)

            # Store other return values
            new_kv_tuples = new_kv_tuples + (new_kv,)

        # Pass linear transformations
        output = self.lin_collapse(self.gelu(self.lin_expand(target)))
        target = target + output
        target = self.norm_out(target)

        # On training, we don't need to cache KV, so clear it.
        new_kv_tuples = None if self.training else new_kv_tuples

        return target, new_kv_tuples


__all__ = ['MultiheadAttentionWeights', 'MultiheadAttention', 'TransformerLayer']
