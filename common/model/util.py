from typing import List, Any

import torch
from torch.nn.functional import pad


def extend_tensor(tensor: torch.Tensor, shape: List[int], fill_value: Any) -> torch.Tensor:
    shape_diff = [(0, max(new - orig, 0))  # (Begin, End)
                  for orig, new in zip(tensor.shape, shape)]
    # pad() accepts padded shape in reversed order of dimensions
    pad_shape = sum(reversed(shape_diff), tuple())
    return pad(tensor, pad_shape, 'constant', fill_value)


def concat_tensors(tensors: List[torch.Tensor], pad_value: Any, dim: int = 0):
    shapes = zip(*[t.shape for t in tensors])
    concat_shape = [max(shapes) for shapes in shapes]
    concat_shape[dim] = -1

    return torch.cat([extend_tensor(tensor, concat_shape, pad_value)
                      for tensor in tensors], dim=dim).contiguous()
