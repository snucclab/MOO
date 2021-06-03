from typing import Dict, Union

import torch
from torch import nn
from torch.nn import functional as F

from common.model.const import PAD_ID, NEG_INF_SAFE, POS_INF_SAFE
from .attention import MultiheadAttention, MultiheadAttentionWeights


def apply_module_dict(modules: nn.ModuleDict, encoded: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Predict next entry using given module and equation.

    :param nn.ModuleDict modules:
        Dictionary of modules to be applied. Modules will be applied with ascending order of keys.
        We expect three types of modules: nn.Linear, nn.LayerNorm and MultiheadAttention.
    :param torch.Tensor encoded:
        Float Tensor that represents encoded vectors.
        Shape [B, T, H], where B = batch size, T = length of equation, and H = hidden dimension.
    :keyword torch.Tensor key_value:
        Float Tensor that represents key and value vectors when computing attention.
        Shape [B, K, H], where K = length of keys
    :keyword torch.Tensor key_ignorance_mask:
        Bool Tensor whose True values at (b, k) make attention layer ignore k-th key on b-th item in the batch.
        Shape [B, K].
    :keyword attention_mask:
        Bool Tensor whose True values at (t, k) make attention layer ignore k-th key when computing t-th query.
        Shape [T, K].
    :rtype: torch.Tensor
    :return:
        Float Tensor that indicates the scores under given information. Shape will be [B, T, ?]
    """
    output = encoded
    keys = sorted(modules.keys())

    # Apply modules (ascending order of keys).
    for key in keys:
        layer = modules[key]
        if isinstance(layer, (MultiheadAttention, MultiheadAttentionWeights)):
            # We will use the first result only (ignore caching)
            output = layer(query=output, **kwargs)[0]
        else:
            output = layer(output)

    return output


def apply_across_dim(function, dim=1, shared_keys=None, **tensors) -> Dict[str, torch.Tensor]:
    """
    Apply a function repeatedly for each tensor slice through the given dimension.
    For example, we have tensor [B, X, S] and dim = 1, then we will concatenate the following matrices on dim=1.
    - function([:, 0, :])
    - function([:, 1, :])
    - ...
    - function([:, X-1, :]).

    :param function: Function to apply.
    :param int dim: Dimension through which we'll apply function. (1 by default)
    :param set shared_keys: Set of keys representing tensors to be shared. (None by default)
    :param torch.Tensor tensors: Keyword arguments of tensors to compute. Dimension should >= `dim`.
    :rtype: Dict[str, torch.Tensor]
    :return: Dictionary of tensors, whose keys are corresponding to the output of the function.
    """
    # Separate shared and non-shared tensors
    shared_arguments = {}
    repeat_targets = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or (shared_keys and key in shared_keys):
            shared_arguments[key] = tensor
        else:
            repeat_targets[key] = tensor

    # Check whether the size of the given dimension is the same across sliced_tensors.
    size = {key: tensor.shape[dim] for key, tensor in repeat_targets.items()}
    assert len(set(size.values())) == 1, 'Tensors does not have same size on dimension %s: We found %s' % (dim, size)

    # Since the sizes are the same, we will represent the size using the first entry.
    size = list(size.values())[0]

    # Dictionary for storing outputs
    output = {}

    for i in range(size):
        # Build kwargs for the function.
        kwargs = {key: tensor.select(dim=dim, index=i).contiguous() for key, tensor in repeat_targets.items()}
        kwargs.update(shared_arguments)

        # Apply function on the slice and restore the dimension for concatenation.
        for key, tensor in function(**kwargs).items():
            if key in shared_keys:
                continue

            if key not in output:
                output[key] = []

            output[key].append(tensor.unsqueeze(dim=dim))

    # Check whether the outputs are have the same size.
    assert all(len(t) == size for t in output.values())

    # Concatenate all outputs, and return.
    return {key: torch.cat(tensor, dim=dim).contiguous() for key, tensor in output.items()}


def get_embedding_without_pad(embedding: Union[nn.Embedding, torch.Tensor],
                              tokens: torch.Tensor, ignore_index=PAD_ID) -> torch.Tensor:
    """
    Get embedding vectors of given token tensor with ignored indices are zero-filled.

    :param nn.Embedding embedding: An embedding instance
    :param torch.Tensor tokens: A Long Tensor to build embedding vectors.
    :param int ignore_index: Index to be ignored. `PAD_ID` by default.
    :rtype: torch.Tensor
    :return: Embedding vector of given token tensor.
    """
    # Clone tokens and fill masked values as zeros.
    tokens = tokens.clone()
    ignore_positions = tokens.eq(ignore_index)
    if ignore_positions.any():
        tokens.masked_fill_(ignore_positions, 0)

    # Apply embedding matrix
    if isinstance(embedding, nn.Embedding):
        embedding = embedding(tokens)
    else:
        embedding = F.embedding(tokens, embedding)

    # Set masked values as zero vector.
    if ignore_positions.any():
        embedding.masked_fill_(ignore_positions.unsqueeze(-1), 0.0)

    return embedding.contiguous()


def init_weights(module: nn.Module, init_factor: float = 0.02):
    """
    Initialize weights

    :param nn.Module module: Module to be initialized.
    """
    if isinstance(module, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
        # nn.Linear has 'weight' and 'bias', nn.Embedding has 'weight',
        # and nn.MultiheadAttention has *_weight and *_bias
        for name, param in module.named_parameters():
            if param is None:
                continue

            if 'weight' in name:
                param.data.normal_(mean=0.0, std=init_factor)
            elif 'bias' in name:
                param.data.zero_()
            else:
                raise NotImplementedError("This case is not considered!")
    elif isinstance(module, nn.LayerNorm):
        # Initialize layer normalization as an identity funciton.
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def is_finite(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(tensor) & (tensor > NEG_INF_SAFE) & (tensor < POS_INF_SAFE)


def logsoftmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute log(softmax(tensor))

    :param torch.Tensor tensor: FloatTensor whose log-softmax value will be computed
    :rtype: torch.FloatTensor
    :return: LogSoftmax result.
    """
    if 0 in tensor.shape:
        # If the tensor is empty, just return it
        return tensor

    # Find maximum values
    max_t = tensor.max(dim=-1, keepdim=True).values
    # Reset maximum as zero if it is not a finite value.
    tensor = tensor - max_t.masked_fill(~is_finite(max_t), 0.0)

    # If a row's elements are all infinity, set the row as zeros to avoid NaN.
    # type: torch.Tensor
    all_inf_mask = (~is_finite(tensor)).all(dim=-1, keepdim=True)
    if all_inf_mask.any().item():
        tensor = tensor.masked_fill(all_inf_mask, 0.0)

    # Forward nn.LogSoftmax.
    return tensor.log_softmax(dim=-1)


class Squeeze(nn.Module):
    """
    Layer class for squeezing a dimension
    """

    def __init__(self, dim: int = -1):
        """
        Layer class for squeezing a dimension

        :param int dim: Dimension to be squeezed, -1 by default.
        """
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Do squeezing

        :param torch.Tensor tensor: FloatTensor to be squeezed
        :rtype: torch.FloatTensor
        :return: Squeezed result
        """
        return tensor.squeeze(dim=self.dim)

    def extra_repr(self):
        # Extra representation for repr()
        return 'dim={dim}'.format(**self.__dict__)


def mask_forward(sz: int, diagonal: int = 1) -> torch.Tensor:
    """
    Generate a mask that ignores future words. Each (i, j)-entry will be True if j >= i + diagonal

    :param int sz: Length of the sequence.
    :param int diagonal: Amount of shift for diagonal entries.
    :rtype: torch.Tensor
    :return: Mask tensor with shape [sz, sz].
    """
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal).contiguous()


def tie_lm_head_with_embed(head: nn.Linear, embedding: nn.Embedding):
    head.in_features = embedding.embedding_dim
    head.out_features = embedding.num_embeddings
    head.weight = embedding.weight
    head.bias.data = F.pad(
        head.bias.data,
        (
            0,
            head.weight.shape[0] - head.bias.shape[0],
        ),
        "constant",
        0,
    )
