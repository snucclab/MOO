from typing import Optional, Callable

from common.const.pad import NEG_INF
from common.data.base import *
from common.model.util import stack_tensors, concat_tensors


class Prediction(TypeTensorBatchable, TypeSelectable):
    #: Log probability vector after the last feed-forward layer. Shape: [T, X] if not batched, else [B, T, X].
    log_prob: torch.Tensor

    def __init__(self, log_prob: torch.Tensor):
        super().__init__()
        self.log_prob = log_prob

    @property
    def shape(self) -> torch.Size:
        return self.log_prob.shape[:-1]

    @property
    def topmost_index(self) -> torch.LongTensor:
        return self.log_prob.argmax(dim=-1)

    @property
    def sequence_lengths(self) -> torch.Tensor:
        size = self.shape[-1]
        return torch.full(self.shape[:-1], fill_value=size, dtype=torch.long)

    @property
    def device(self) -> torch.device:
        return self.log_prob.device

    @classmethod
    def build_batch(cls, *items: 'Prediction') -> 'Prediction':
        return Prediction(stack_tensors([item.log_prob for item in items], pad_value=NEG_INF))

    @classmethod
    def concat(cls, *items: 'Prediction') -> 'Prediction':
        return Prediction(concat_tensors([item.log_prob for item in items], pad_value=NEG_INF))

    def as_dict(self) -> dict:
        return dict(log_prob=self.log_prob)

    def topk(self, k: int):
        return self.log_prob.topk(k=k, dim=-1)

    def to_human_readable(self, converter: Optional[Callable[[int], str]] = None) -> dict:
        result = dict(shape=list(self.shape))
        if converter is None:
            result['prediction'] = human_readable_form(self.topmost_index)
        else:
            predicted = self.topmost_index.tolist()
            if self.is_batched:
                result['prediction'] = [' '.join([converter(t) for t in row])
                                        for row in predicted]
            else:
                result['prediction'] = ' '.join([converter(t) for t in predicted])

        return result
