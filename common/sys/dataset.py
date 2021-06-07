from pathlib import Path
from time import sleep
from typing import List, Dict, Set

import torch
from numpy import mean, std
from numpy.random import Generator, PCG64

from common.model.const import DEF_ENCODER, PAD_ID
from common.model.types import Text, Expression
from common.solver.types import Execution
from common.solver.const import OPR_MAX_ARITY, OPR_NEW_EQN_ID, CON_MAX
from .convert import string_to_text_instance
from .key import ITEM_ID, QUESTION, EXECUTION, ANSWER, SRC_CONSTANT, SRC_NUMBER

CACHE_ITEMS = 'item'
CACHE_TOKENIZER = 'tokenizer'
CACHE_VOCAB_SZ = 'vocab_size'


def _word_counts(indices, tokenizer):
    single_dim = len(indices.shape) == 1
    if single_dim:
        indices = indices.unsqueeze(0)

    indices = indices.pad_fill(tokenizer.pad_token_id).tolist()
    text = tokenizer.batch_decode(indices, skip_special_tokens=True)
    text = [len(line.split(' ')) for line in text]

    if single_dim:
        return text[0]
    else:
        return text


def _get_stats_of(values: List[int]) -> dict:
    return {
        'min': min(values),
        'mean': float(mean(values)),
        'stdev': float(std(values)),
        'max': max(values),
        'N': len(values)
    }


class Example:
    def __init__(self, item_id: str, text: Text, execution: List[Execution], answer: str):
        self.item_id = item_id
        self.text = text
        self.execution = execution
        self.answer = answer

    @classmethod
    def from_dict(cls, item: dict, tokenizer) -> 'Example':
        return cls(item_id=item[ITEM_ID],
                   text=string_to_text_instance(item[QUESTION], tokenizer),
                   execution=[Execution.from_list(x) for x in item[EXECUTION]],
                   answer=item[ANSWER])

    def get_item_size(self) -> int:
        return max(self.text.shape[-1], len(self.execution))


class BatchedExample:
    def __init__(self, items: List[Example]):
        self.item_id = [x.item_id for x in items]
        self.text = Text.build_batch(*[x.text for x in items])
        self.answer = [x.answer for x in items]

        # Convert execution into expression
        word_size = max(len(x) for x in self.text.word_info)
        expressions = []
        for item in items:
            functions = [OPR_NEW_EQN_ID]
            operands = [[PAD_ID] for _ in range(OPR_MAX_ARITY)]

            for x in item.execution:
                functions.append(x.function)
                for j, (src, a_j) in enumerate(x.arguments):
                    if src == SRC_CONSTANT:
                        operands[j].append(a_j)
                    elif src == SRC_NUMBER:
                        operands[j].append(a_j + CON_MAX)
                    else:
                        operands[j].append(a_j + CON_MAX + word_size)
            expressions.append(Expression(operator=torch.tensor([functions], dtype=torch.long),
                                          operands=[torch.tensor([o], dtype=torch.long) for o in operands]))

        # Build batch of expressions
        self.expression = Expression.build_batch(*expressions)


class Dataset:
    def __init__(self, path: str, langmodel: str = DEF_ENCODER, seed: int = 1):
        from transformers import AutoTokenizer

        # List of problem items
        self._whole_items: List[Example] = []
        # List of selected items
        self._items: List[Example] = []
        # Map from experiments to set of ids (to manage splits)
        self._split_map: Dict[str, Set[str]] = {}
        # Vocab size of the tokenizer
        self._vocab_size: int = 0
        # Path of this dataset
        self._path = path
        # Lang Model applied in this dataset
        self._langmodel = langmodel
        self._tokenizer = AutoTokenizer.from_pretrained(self._langmodel)
        # RNG for shuffling
        self._rng = Generator(PCG64(seed))

        # Read the dataset.
        cache_loaded = self._try_read_cache()
        if not cache_loaded:
            self._whole_items = self._save_cache()

    def _save_cache(self) -> List[Example]:
        # Otherwise, compute preprocessed result and cache it in the disk
        from json import load
        from torch import save

        # Make lock file
        save(True, str(self.cache_lock_path))

        # First, read the JSON with lines file.
        self._vocab_size = len(self._tokenizer)
        with Path(self._path).open('r+t', encoding='UTF-8') as fp:
            items = [Example.from_dict(item, self._tokenizer)
                     for item in load(fp)]

        # Cache dataset and vocabulary.
        save({
            CACHE_TOKENIZER: self._langmodel,
            CACHE_ITEMS: items,
            CACHE_VOCAB_SZ: self._vocab_size
        }, str(self.cached_path))

        # Delete lock file
        self.cache_lock_path.unlink()
        return items

    def _try_read_cache(self):
        while self.cache_lock_path.exists():
            # Wait until lock file removed (sleep 0.1s)
            sleep(0.1)

        if self.cached_path.exists():
            # If cached version is available, load the dataset from it.
            from torch import load
            cache = load(self.cached_path)

            if self._langmodel == cache[CACHE_TOKENIZER]:
                self._whole_items = cache[CACHE_ITEMS]
                self._vocab_size = cache[CACHE_VOCAB_SZ]
                return True

        return False

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def cached_path(self) -> Path:
        return Path(f'{self._path}.cached')

    @property
    def cache_lock_path(self) -> Path:
        return Path(f'{self._path}.lock')

    @property
    def num_items(self) -> int:
        return len(self._items)

    @property
    def get_dataset_name(self) -> str:
        return Path(self._path).stem

    @property
    def statistics(self) -> Dict[str, float]:
        return self.get_statistics()

    def get_statistics(self, as_whole: bool = True) -> Dict[str, float]:
        item_list = self._whole_items if as_whole else self._items
        return {
            'items': len(item_list),
            'text.tokens': _get_stats_of([item.text.sequence_lengths.item()
                                          for item in item_list]),
            'text.words': _get_stats_of([item.text.word_indexes.max().item()
                                         for item in item_list]),
            'expression': _get_stats_of([len(item.execution)
                                         for item in item_list])
        }

    def get_rng_state(self):
        return self._rng.__getstate__()

    def set_rng_state(self, state):
        self._rng.__setstate__(state)

    def reset_seed(self, seed):
        self._rng = Generator(PCG64(seed))

    def select_items_in(self, ids: Set[str]):
        # Filter items and build task groups
        self._items = [item
                       for item in self._whole_items
                       if item.item_id in ids]

    def select_items_with_file(self, path: str):
        if path not in self._split_map:
            with Path(path).open('rt') as fp:
                self._split_map[path] = set([line.strip() for line in fp.readlines()])

        self.select_items_in(self._split_map[path])

    def get_minibatches(self, batch_size: int = 8):
        chunk_step = batch_size * 10
        item_in_a_batch = []

        items = self._items.copy()
        self._rng.shuffle(items)

        for begin in range(0, len(items), chunk_step):
            chunk = items[begin:begin + chunk_step]
            for item in sorted(chunk, key=Example.get_item_size):
                item_in_a_batch.append(item)

                if len(item_in_a_batch) == batch_size:
                    yield BatchedExample(item_in_a_batch)
                    item_in_a_batch.clear()

        if item_in_a_batch:
            yield BatchedExample(item_in_a_batch)
