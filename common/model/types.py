import abc
from typing import TypeVar, Any, ItemsView

import torch

TypedAny = TypeVar('TypedAny')


def human_readable_form(value: TypedAny, **kwargs):
    try_readable = getattr(value, 'to_human_readable', None)
    if callable(try_readable):
        return try_readable(**kwargs)
    elif isinstance(value, TypeBase):
        return human_readable_form(value.as_dict(), **kwargs)
    elif isinstance(value, dict):
        return {k: human_readable_form(v, **kwargs) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [human_readable_form(v, **kwargs) for v in value]
    elif isinstance(value, (int, float, str)):
        return str(value)
    elif isinstance(value, torch.Tensor):
        entries = {'': value.tolist()}
        dim = value.dim()

        while dim > 1:
            new_entry = {}
            for index_prefix, tensor_list in entries.items():
                new_entry.update({
                    f'{index_prefix}.{row}': tensor
                    for row, tensor in enumerate(tensor_list)
                })
            dim -= 1
            entries = new_entry

        return {
            # Remove the first '.'
            key[1:]: '\t'.join([str(x) for x in tensor])
            for key, tensor in entries.items()
        }
    else:
        return value


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
            if type(value) is list and isinstance(value[0], TypeSelectable):
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

    @abc.abstractmethod
    def to_human_readable(self, **kwargs) -> dict:
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
