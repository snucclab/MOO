from typing import Callable, Optional

import torch

from common.model.util import stack_tensors, concat_tensors
from common.model.base import TypeTensorBatchable, TypeSelectable


class Encoded(TypeTensorBatchable, TypeSelectable):
    vector: torch.Tensor
    pad: torch.Tensor

    def __init__(self, vector: torch.Tensor, pad: Optional[torch.Tensor]):
        super().__init__()
        if pad is None:
            pad = torch.zeros(vector.shape[:-1], dtype=torch.bool, device=vector.device)

        assert vector.shape[:-1] == pad.shape
        self.vector = vector
        self.pad = pad
    
    def __add__(self, other: 'Encoded') -> 'Encoded':
        assert self.vector.shape == other.vector.shape
        return Encoded(self.vector + other.vector, self.pad)

    def __mul__(self, other: float) -> 'Encoded':
        return Encoded(self.vector * other, self.pad)

    @property
    def shape(self) -> torch.Size:
        return self.pad.shape

    @property
    def device(self) -> torch.device:
        return self.vector.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.pad.logical_not().sum(dim=-1)

    @property
    def attn_mask_float(self) -> torch.Tensor:
        return self.pad.logical_not().float()

    @property
    def pooled_state(self) -> 'Encoded':
        sum_of_states = self.vector.masked_fill(self.pad.unsqueeze(-1), 0).sum(dim=-2)
        len_of_states = self.sequence_lengths
        pooled = sum_of_states / len_of_states.unsqueeze(-1)
        pooled_pad = len_of_states.eq(0)
        return Encoded(pooled, pooled_pad)

    @classmethod
    def build_batch(cls, *items: 'Encoded') -> 'Encoded':
        vectors = stack_tensors([item.vector for item in items], pad_value=0.0)
        pads = stack_tensors([item.pad for item in items], pad_value=True)
        return Encoded(vectors, pads)

    @classmethod
    def concat(cls, *items: 'Encoded', dim: int = 0) -> 'Encoded':
        vectors = concat_tensors([item.vector for item in items], dim=dim, pad_value=0.0)
        pads = concat_tensors([item.pad for item in items], dim=dim, pad_value=True)
        return Encoded(vectors, pads)

    @classmethod
    def empty(cls, *shape: int, device='cpu') -> 'Encoded':
        return Encoded(torch.empty(*shape, device=device),
                       torch.empty(*shape[:-1], dtype=torch.bool, device=device))

    def as_dict(self) -> dict:
        return dict(vector=self.vector, pad=self.pad)

    def repeat(self, n: int) -> 'Encoded':
        return Encoded(vector=self.vector.expand((n,) + self.vector.shape[1:]),
                       pad=self.pad.expand((n,) + self.pad.shape[1:]))

    def unsqueeze(self, dim: int) -> 'Encoded':
        return Encoded(self.vector.unsqueeze(dim), self.pad.unsqueeze(dim))

    def pad_fill(self, fill_value: float) -> torch.Tensor:
        return self.vector.masked_fill(self.pad.unsqueeze(-1), fill_value)

    def apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> 'Encoded':
        new_vector = fn(self.vector)
        assert new_vector.shape[:-1] == self.shape

        return Encoded(new_vector, self.pad)

    def to_human_readable(self, **kwargs) -> dict:
        return {
            'shape': self.shape
        }

    #
    # def select(self, index: int) -> 'Encoded':
    #     return Encoded(self.vector[index], self.pad[index])
    #
    # def unsqueeze(self, dim: int) -> 'Encoded':
    #     return Encoded(self.vector.unsqueeze(dim), self.pad.unsqueeze(dim))
    #
    # def detach(self) -> 'Encoded':
    #     return Encoded(self.vector.detach(), self.pad.detach())
    #
    # def slice(self, begin: int = 0, end: int = None) -> 'Encoded':
    #     end = self.pad.shape[1] if end is None else end
    #     return Encoded(self.vector[:, begin:end], self.pad[:, begin:end])
    #
    # def reduce(self, reduction: str = 'mean', dim: int = 0) -> torch.Tensor:
    #     if reduction == 'first':
    #         return self.vector.select(dim=dim, index=0)
    #     if reduction == 'last':
    #         if self.pad is not None:
    #             position_shape = list(self.vector.shape)
    #             position_shape[dim] = 1
    #
    #             last_pos = (~self.pad).long().sum(dim=dim, keepdim=True) - 1
    #             last_pos = last_pos.unsqueeze(-1).expand(*position_shape)
    #
    #             return self.vector.gather(dim=dim, index=last_pos)
    #         else:
    #             return self.vector.select(dim=dim, index=-1)
    #     if reduction == 'mean':
    #         if self.pad is not None:
    #             return self.vector.sum(dim=dim) / (~self.pad).float().sum(dim=dim, keepdim=True)
    #         else:
    #             return self.vector.mean(dim=dim)
    #     if reduction == 'sum':
    #         return self.vector.sum(dim=dim)
    #     if reduction == 'max':
    #         return self.vector.max(dim=dim).values
    #
    #     raise ValueError('"%s" is not a supported reduction method!' % reduction)
    #
    # @staticmethod
    # def zeros(*size, device=None) -> 'Encoded':
    #     encoded = torch.zeros(size, device=device)
    #     pad = torch.ones(size, dtype=torch.bool, device=device)
    #     return Encoded(encoded, pad)
    #
    # def concatenate(self, other: 'Encoded', dim: int = 1) -> 'Encoded':
    #     return Encoded(torch.cat([self.vector, other.vector], dim=dim).contiguous(),
    #                    torch.cat([self.pad, other.pad], dim=dim).contiguous())
    #
    # def masked_select(self, var_position: torch.Tensor) -> 'Encoded':
    #     new_pad: torch.Tensor = self.pad | ~var_position.to(self.pad.device)
    #     max_pad = (~new_pad).sum(dim=-1).max().item()
    #
    #     new_pad = new_pad[:, :max_pad]
    #     new_encoded = self.vector[:, :max_pad].masked_fill(new_pad.unsqueeze(-1), 0.0)
    #     return Encoded(new_encoded, new_pad)
