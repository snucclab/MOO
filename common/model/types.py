import abc
from typing import TypeVar, Any, ItemsView, List, Dict, Optional, Callable

import torch

from common.model.const import PAD_ID
from common.model.util import stack_tensors, concat_tensors
from common.solver.const import OPR_MAX_ARITY, OPR_NEW_EQN_ID

TypedAny = TypeVar('TypedAny')


def move_to(value: TypedAny, *args, **kwargs) -> TypedAny:
    if isinstance(value, torch.Tensor):
        return value.to(*args, **kwargs)
    elif isinstance(value, TypeBase):
        cls = value.__class__
        return cls(**move_to(value.as_dict(), *args, **kwargs))
    elif isinstance(value, dict):
        return {k: move_to(v, *args, **kwargs) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [move_to(v, *args, **kwargs) for v in value]
    else:
        return value


class TypeBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()

    def as_tuples(self) -> ItemsView[str, Any]:
        return self.as_dict().items()

    def copy(self, **update):
        cls = self.__class__
        kwargs = self.as_dict()
        kwargs.update({key: update[key]
                       for key in kwargs if key in update})
        return cls(**kwargs)

    def to(self, *args, **kwargs):
        return move_to(self, *args, **kwargs)


class TypeSelectable(TypeBase, abc.ABC):
    def __getitem__(self, item):
        cls = self.__class__
        kwargs = {}
        for key, value in self.as_dict().items():
            if type(value) is list and isinstance(value[0], (torch.Tensor, TypeSelectable)):
                kwargs[key] = [v[item] for v in value]
            elif isinstance(value, (torch.Tensor, TypeSelectable)):
                kwargs[key] = value[item]
            else:
                kwargs[key] = value

        return cls(**kwargs)


class TypeBatchable(TypeBase, abc.ABC):
    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_batched(self) -> bool:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def build_batch(cls, *items):
        raise NotImplementedError()


class TypeTensorBatchable(TypeBatchable):
    @property
    @abc.abstractmethod
    def shape(self) -> torch.Size:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def sequence_lengths(self) -> torch.LongTensor:
        raise NotImplementedError()

    @property
    def is_batched(self) -> bool:
        return len(self.shape) > 1


class Text(TypeTensorBatchable, TypeSelectable):
    #: 토큰 인덱스. 항상 [B, S], B는 Batch size, S는 토큰 길이(가변)
    tokens: torch.LongTensor
    #: 토큰 별 단어 인덱스. tokens와 길이 동일. common.model.const.PAD_ID 값은 단어가 아닌 경우.
    word_indexes: torch.LongTensor
    #: 각 단어에 관한 정보. len(word_info) == B, len(word_info[i]) == word_indexs[i].max().
    word_info: List[List[Dict[str, Any]]]
    #   'is_num'(common.sys.key.IS_NUM)은 단어가 십진숫자인지의 여부 (True/False)
    #   'is_var'(common.sys.key.IS_VAR)는 단어가 미지수[A, B, C, ..., Z]인지의 여부
    #   'is_prop'(common.sys.key.IS_PROP)는 단어가 고유명사[(가), ... 정국, ...]인지의 여부
    #   'value'(common.sys.key.VALUE)는 단어의 불필요한 부분을 제외한 어근의 값 (string)
    #   'word'(common.sys.key.WORD)는 단어 자체

    def __init__(self, tokens: torch.LongTensor, word_indexes: torch.LongTensor, word_info: List[List[Dict[str, Any]]]):
        assert tokens.dim() == 2
        assert tokens.shape == word_indexes.shape
        assert tokens.shape[0] == len(word_info)

        super().__init__()
        self.tokens = tokens
        self.word_indexes = word_indexes
        self.word_info = word_info

    @property
    def shape(self) -> torch.Size:
        return self.tokens.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.tokens.eq(PAD_ID)

    @property
    def attn_mask_float(self) -> torch.Tensor:
        return self.pad.logical_not().float()

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.pad.logical_not().sum(dim=-1)

    @classmethod
    def build_batch(cls, *items: 'Text') -> 'Text':
        return Text(tokens=concat_tensors([item.tokens for item in items], pad_value=PAD_ID),
                    word_indexes=concat_tensors([item.word_indexes for item in items], pad_value=PAD_ID),
                    word_info=[info for item in items for info in item.word_info])

    def tokens_pad_fill(self, fill_value: int) -> torch.Tensor:
        return self.tokens.masked_fill(self.pad, fill_value)

    def as_dict(self) -> dict:
        return dict(tokens=self.tokens, word_indexes=self.word_indexes, word_info=self.word_info)


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


class ExpressionPrediction(TypeSelectable):
    #: Operator prediction
    operator: torch.Tensor
    #: List of operand predictions, [operand0, operand1, ...]. Each item is either batched or not.
    operands: List[torch.Tensor]

    def __init__(self, operator: torch.Tensor, operands: List[torch.Tensor]):
        super().__init__()
        assert all(operator.shape[:-1] == operand_j.shape[:-1] for operand_j in operands), \
            "%s vs %s" % (operator.shape, [operand_j.shape for operand_j in operands])
        assert len(operands) == OPR_MAX_ARITY
        assert all(operand_j.dim() == operator.dim() for operand_j in operands)
        self.operator = operator
        self.operands = operands

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)


class Expression(TypeTensorBatchable, TypeSelectable):
    #: 연산자/함수명 인덱스. 항상 [B, T], B는 Batch size, T는 토큰 길이(가변).
    operator: torch.LongTensor
    #: 피연산자의 목록. operator와 길이 동일. common.model.const.PAD_ID는 정의되지 않은 경우.
    operands: List[torch.LongTensor]

    def __init__(self, operator: torch.LongTensor, operands: List[torch.LongTensor]):
        super().__init__()
        assert all(operator.shape == operand_j.shape for operand_j in operands), \
            "%s vs %s" % (operator.shape, [operand_j.shape for operand_j in operands])
        assert len(operands) == OPR_MAX_ARITY
        assert operator.dim() == 2 and all(operand_j.dim() == 2 for operand_j in operands)
        self.operator = operator
        self.operands = operands

    @property
    def shape(self) -> torch.Size:
        return self.operator.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.operator.eq(PAD_ID)

    @property
    def device(self) -> torch.device:
        return self.operator.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.pad.logical_not().sum(dim=-1)

    @classmethod
    def build_batch(cls, *items: 'Expression') -> 'Expression':
        operands = zip(*[item.operands for item in items])
        return Expression(operator=concat_tensors([item.operator for item in items], pad_value=PAD_ID),
                          operands=[concat_tensors(operand_j, pad_value=PAD_ID) for operand_j in operands])

    @classmethod
    def get_generation_base(cls) -> 'Expression':
        operator = torch.LongTensor([[OPR_NEW_EQN_ID]])  # [M=1, T=1]
        operands = [torch.LongTensor([[PAD_ID]])  # [M=1, T=1]
                    for _ in range(OPR_MAX_ARITY)]
        return Expression(operator, operands)

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)

    def extends_to(self, next_operator: torch.LongTensor, next_operands: List[torch.LongTensor]) -> 'Expression':
        return Expression(operator=torch.cat([self.operator, next_operator], dim=-1).contiguous(),
                          operands=[torch.cat([prev_j, next_j], dim=-1).contiguous()
                                    for prev_j, next_j in zip(self.operands, next_operands)])

    def unsqueeze(self, dim: int) -> 'Expression':
        return Expression(operator=self.operator.unsqueeze(dim),
                          operands=[operand_j.unsqueeze(dim)
                                    for operand_j in self.operands])
