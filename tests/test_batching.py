import pytest
import torch

from common.model.const import DEF_ENCODER
from common.model.types import Text, Expression
from common.solver.const import OPR_MAX_ARITY, CON_MAX
from common.sys.const import MODULE_RSC_PATH
from common.sys.convert import equation_to_execution
from common.sys.dataset import Dataset


@pytest.fixture(scope='session')
def dataset():
    dataset = Dataset(path=MODULE_RSC_PATH / 'dataset' / 'dataset.json', langmodel=DEF_ENCODER, seed=1)
    yield dataset


def test_build_batch(dataset):
    for batch in dataset.get_minibatches(8):
        bsz = len(batch.item_id)
        raw_items = [dataset._get_raw_item(i) for i in batch.item_id]

        # Item ID type check
        assert all(type(i) is str for i in batch.item_id)

        # Answer type check
        assert all(type(i) is str and i == i.strip() for i in batch.answer)
        assert len(batch.answer) == bsz

        # Answer value check
        assert all(i == r.answer for i, r in zip(batch.answer, raw_items))

        # Text type check
        assert isinstance(batch.text, Text)
        assert batch.text.shape[0] == bsz
        assert len(batch.text.shape) == 2
        assert batch.text.shape == batch.word_indexes.shape
        words = {}
        for i in range(bsz):
            assert batch.text.word_indexes[i].max().item() == len(batch.text.word_info[i]) - 1
            assert all(type(i) is dict for i in batch.text.word_info[i])
            words[i] = len(batch.text.word_info[i])

            # Text value check
            _, text_len = raw_items[i].text.shape
            assert batch.text.word_indexes[i:i+1, text_len] == raw_items[i].text.word_indexes
            assert batch.text.tokens[i:i+1, text_len] == raw_items[i].text.tokens
            assert batch.text.word_info[i] == raw_items[i].text.word_info

        # Expression type check
        assert isinstance(batch.expression, Expression)
        assert batch.expression.shape[0] == bsz
        assert len(batch.expression.operands) == OPR_MAX_ARITY
        max_word = max(words.values())
        res_begin = CON_MAX + max_word

        # Expression value check
        argument_thr = torch.arange(batch.expression.shape[-1]) - 1 + res_begin

        for i in range(bsz):
            expr = batch.expression[i]
            assert all(arg.lt(argument_thr).all() for arg in expr.operands)
            assert not any(arg.lt(res_begin).logical_and(arg.ge(CON_MAX + words[i])).any() for arg in expr.operands)
