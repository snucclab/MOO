from typing import Union, List, Optional, Callable

from common.const.pad import PAD_ID
from common.model.loss import SmoothedCrossEntropyLoss
from common.model.util import stack_tensors, concat_tensors
from common.model.base import *
from .prediction import Prediction

CROSS_ENTROPY_LOSS = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
SMOOTHED_CROSS_ENTROPY_LOSS = SmoothedCrossEntropyLoss(ignore_index=PAD_ID)


class Label(TypeTensorBatchable, TypeSelectable):
    #: Long Tensor of gold target instance. Shape: [T] if not batched else [B, T].
    indices: torch.Tensor

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.indices = indices

    def __repr__(self) -> str:
        return 'Label(%s)' % self.indices.tolist()

    @property
    def shape(self) -> torch.Size:
        return self.indices.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.indices.eq(PAD_ID)

    @property
    def attn_mask_float(self) -> torch.Tensor:
        return self.pad.logical_not().float()

    @property
    def device(self) -> torch.device:
        return self.indices.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.pad.logical_not().sum(dim=-1)

    @property
    def shifted_indices(self) -> torch.LongTensor:
        slicing = [slice(None) for _ in range(self.indices.dim() - 1)] + [slice(1, None)]
        return self.indices[tuple(slicing)]

    @classmethod
    def build_batch(cls, *items: 'Label') -> 'Label':
        return Label(stack_tensors([item.indices for item in items], pad_value=PAD_ID))

    @classmethod
    def concat(cls, *items: 'Label', dim: int = 0) -> 'Label':
        return Label(concat_tensors([item.indices for item in items], dim=dim, pad_value=PAD_ID))

    @classmethod
    def from_list(cls, items: Union[List[int], List[List[int]]]) -> 'Label':
        if type(items) is list and type(items[0]) is list:
            items = stack_tensors([torch.LongTensor(item) for item in items], pad_value=PAD_ID)
        else:
            items = torch.LongTensor(items)

        return Label(items)

    @classmethod
    def empty(cls, batch_sz: int) -> 'Label':
        return Label(torch.full((batch_sz, 0), fill_value=PAD_ID, dtype=torch.long))

    def as_dict(self) -> dict:
        return dict(indices=self.indices)

    def pad_fill(self, fill_value: int) -> torch.Tensor:
        return self.indices.masked_fill(self.pad, fill_value)

    def smoothed_cross_entropy(self, output: Prediction, smoothing: float = 0.1, focal: float = 0.0,
                               shift_target: bool = True) -> torch.Tensor:
        assert output.shape == self.shape

        if shift_target:
            target = self.shifted_indices

            slicing = [slice(None) for _ in range(self.indices.dim() - 1)] + [slice(-1)]
            output = output.log_prob[tuple(slicing)]
        else:
            target = self.indices
            output = output.log_prob

        if smoothing == 0:
            return CROSS_ENTROPY_LOSS(output.flatten(end_dim=-2), target.flatten().to(output.device))
        else:
            return SMOOTHED_CROSS_ENTROPY_LOSS(output, target.to(output.device), smoothing=smoothing, focal=focal)

    def num_corrects(self, output: Prediction, shift_target: bool = True) -> dict:
        assert output.shape == self.shape, f'{output.shape} != {self.shape}'
        assert self.is_batched

        if shift_target:
            target = self.shifted_indices.cpu()

            slicing = [slice(None) for _ in range(self.indices.dim() - 1)] + [slice(-1)]
            predict = output.topmost_index[tuple(slicing)].cpu()
        else:
            target = self.indices.cpu()
            predict = output.topmost_index.cpu()

        # Shape [B, T]
        nonpad = target.ne(PAD_ID)

        correct = predict.eq(target)
        token = correct.logical_and(nonpad).float()
        seq = correct.logical_or(nonpad.logical_not())

        return {
            'token': {
                'corrects': float(token.sum()),
                'total': float(nonpad.sum())
            },
            'seq': {
                'corrects': float(seq.prod(dim=1).sum()),
                'total': int(seq.shape[0]),
                'raw': seq
            }
        }

    def accuracy_of(self, output: Prediction, shift_target: bool = True) -> dict:
        counts = self.num_corrects(output, shift_target)
        token = counts['token']
        seq = counts['seq']

        return {
            'token_acc': token['corrects'] / token['total'] if token['total'] else float('NaN'),
            'seq_acc': seq['corrects'] / seq['total'] if seq['total'] else float('NaN'),
            'seq': seq['raw']
        }

    def to_human_readable(self, converter: Optional[Callable[[int], str]] = None) -> dict:
        result = dict(shape=list(self.shape))
        if converter is None:
            result['target'] = human_readable_form(self.indices)
        else:
            predicted = self.indices.tolist()
            if self.is_batched:
                result['target'] = [' '.join([converter(t) for t in row]).strip()
                                    for row in predicted]
            else:
                result['target'] = ' '.join([converter(t) for t in predicted]).strip()

        return result

    def extends_to(self, next_label_token: torch.LongTensor) -> 'Label':
        assert next_label_token.dim() == self.indices.dim()
        assert next_label_token.shape[:-1] == self.indices.shape[:-1]

        return Label(torch.cat([self.indices, next_label_token], dim=-1))

    def prepend(self, prefix: torch.LongTensor, dim: int = -1) -> 'Label':
        assert prefix.dim() == 1

        # Unsqueeze other dimensions
        slices: list = [None] * self.indices.dim()
        slices[dim] = slice(None)
        prefix = prefix[tuple(slices)]

        # Expand prefix to match other dimensions
        expand_shape = list(self.indices.shape)
        expand_shape[dim] = -1
        prefix = prefix.expand(*expand_shape).to(self.indices)

        return Label(torch.cat([prefix, self.indices], dim=dim))

    def flatten(self) -> 'Label':
        return Label(self.indices.masked_select(self.pad.logical_not()))

    def unsqueeze(self, dim: int) -> 'Label':
        return Label(self.indices.unsqueeze(dim))

    def repeat(self, n: int) -> 'Label':
        return Label(indices=self.indices.expand((n,) + self.indices.shape[1:]))

    def ignore_labels(self, excluded: set) -> 'Label':
        indices = self.indices
        for exc in excluded:
            indices = indices.masked_fill(indices.eq(exc), PAD_ID)

        return Label(indices)
