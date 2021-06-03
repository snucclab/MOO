from .encoded import *
from .expression import *
from .label import *
from .prediction import *
from .text import *


L1_LOSS = torch.nn.L1Loss()


def _compute_accuracy_from_list(items: list, key: str = '') -> dict:
    values = {}
    for tgt in {'token', 'seq'}:
        for field in {'corrects', 'total'}:
            values[tgt + field] = sum([item[tgt][field] for item in items])

    return {
        f'token_acc_{key}': values['tokencorrects'] / values['tokentotal'] if values['tokentotal'] else float('NaN'),
        f'seq_acc_{key}': values['seqcorrects'] / values['seqtotal'] if values['seqtotal'] else float('NaN')
    }


class ExtraInfo(TypeBase):
    item_id: str
    answers: List[Dict[str, sympy.Number]]
    numbers: Dict[str, sympy.Number]
    variables: List[str]
    split: Optional[str]
    raw: Optional[dict]

    def __init__(self, item_id: str, answers: List[Dict[str, sympy.Number]], numbers: Dict[str, sympy.Number],
                 variables: List[str], split: str = None, raw: dict = None):
        super().__init__()
        self.item_id = item_id
        self.answers = answers
        self.numbers = numbers
        self.variables = variables
        self.split = split
        self.raw = raw

    @classmethod
    def from_dict(cls, raw: dict) -> 'ExtraInfo':
        answers = [{key: sympy.Number(var)
                    for key, var in item.items() if not key.startswith('_')}
                   for item in raw['answers'] if item['_selected']]
        numbers = {num['key']: sympy.Number(num['value'])
                   for num in raw['numbers']}
        return ExtraInfo(item_id=raw['_id'], split=None, answers=answers, numbers=numbers, variables=[], raw=raw)

    def filter_answers(self) -> 'ExtraInfo':
        kwargs = self.as_dict()
        kwargs['answers'] = [{key: value
                              for key, value in answer_tuple.items()
                              if key in kwargs['variables']}
                             for answer_tuple in kwargs['answers']]
        return ExtraInfo(**kwargs)

    def as_dict(self) -> dict:
        return dict(item_id=self.item_id, answers=self.answers, numbers=self.numbers, variables=self.variables,
                    split=self.split, raw=self.raw)

    def to_human_readable(self) -> dict:
        return human_readable_form(self.as_dict())


class Example(TypeBatchable):
    text: Text
    expression: Expression
    description: Union[Description, List[Description]]
    info: Union[ExtraInfo, List[ExtraInfo]]

    def __init__(self, text: Text, expression: Expression, description: Union[Description, List[Description]],
                 info: Union[ExtraInfo, List[ExtraInfo]]):
        super().__init__()
        self.text = text
        self.expression = expression
        self.description = description
        self.info = info

    @property
    def device(self) -> torch.device:
        return self.text.device

    @property
    def is_batched(self) -> bool:
        return self.text.is_batched

    @property
    def batch_size(self):
        return self.text.shape[0] if self.is_batched else 1

    @classmethod
    def build_batch(cls, *items: 'Example') -> 'Example':
        return Example(text=Text.build_batch(*[item.text for item in items]),
                       expression=Expression.build_batch(*[item.expression for item in items]),
                       description=[item.description for item in items],
                       info=[item.info for item in items])

    @classmethod
    def concat(cls, *items: 'Example') -> 'Example':
        raise NotImplementedError('This operation is not supported')

    @classmethod
    def from_dict(cls, raw: dict, tokenizer, number_window: int = 5, formula: str = 'equations') -> 'Example':
        _info = ExtraInfo.from_dict(raw)
        _text = Text.from_dict(raw, tokenizer=tokenizer, number_window=number_window)
        _expression = Expression.from_dict(raw, var_list_out=_info.variables, field=formula)
        _description = Description.from_dict(raw, n_numbers=len(_info.numbers),
                                             var_list=_info.variables, tokenizer=tokenizer)

        # Filter out not-used variables from the answers
        _info = _info.filter_answers()

        return Example(text=_text, expression=_expression, description=_description, info=_info)

    def as_dict(self) -> dict:
        return dict(text=self.text, expression=self.expression, description=self.description, info=self.info)

    def get_item_size(self) -> int:
        return max(self.text.shape[-1], self.expression.shape[-1], self.description.number_for_train.shape[-1],
                   self.description.variable_for_train.shape[-1])

    def item_of_batch(self, index: int) -> 'Example':
        assert self.is_batched
        return Example(text=self.text[index],
                       expression=self.expression[index],
                       description=self.description[index],
                       info=self.info[index])

    def to_human_readable(self, tokenizer=None) -> dict:
        if self.is_batched:
            return dict(
                info=[i.to_human_readable() for i in self.info],
                text=self.text.to_human_readable(tokenizer),
                expression=self.expression.to_human_readable(),
                description=[d.to_human_readable(tokenizer) for d in self.description]
            )
        else:
            return dict(
                info=self.info.to_human_readable(),
                text=self.text.to_human_readable(tokenizer),
                expression=self.expression.to_human_readable(),
                description=self.description.to_human_readable(tokenizer)
            )

    def accuracy_of(self, **kwargs) -> dict:
        # expression: ExpressionPrediction [B, T]
        # num_desc?: B-List of Prediction [N, D]
        # var_desc?: B-List of Prediction [V, D] or Prediction [B, VD]
        # var_target?: Label [B, VD]
        result = {}
        if 'expression' in kwargs:
            if 'expression_tgt' in kwargs:
                expr_tgt = kwargs['expression_tgt']
            elif 'expr_ignore' in kwargs:
                expr_tgt = self.expression.ignore_labels(kwargs['expr_ignore'])
            else:
                expr_tgt = self.expression
            result.update(expr_tgt.accuracy_of(kwargs['expression']))

        if 'num_desc' in kwargs:
            num_desc_cnt = [gold.number_for_train.num_corrects(pred)
                            for gold, pred in zip(self.description, kwargs['num_desc'])]
            result.update(_compute_accuracy_from_list(num_desc_cnt, key='num'))

        if 'var_desc' in kwargs:
            var_desc_cnt = [gold.variable_for_train.num_corrects(pred)
                            for gold, pred in zip(self.description, kwargs['var_desc'])]
            result.update(_compute_accuracy_from_list(var_desc_cnt, key='var'))

        if 'var_len' in kwargs:
            var_len: torch.Tensor = kwargs['var_len_target'].float()
            var_len_pred: torch.Tensor = kwargs['var_len'].round().float()
            result['diff_var_len'] = (var_len_pred - var_len).mean().item()

        return result

    def smoothed_cross_entropy(self, **kwargs) -> Dict[str, torch.Tensor]:
        # expression: ExpressionPrediction [B, T]
        # num_desc?: B-List of Prediction [N, D]
        # var_desc?: B-List of Prediction [V, D] or Prediction [B, VD]
        # var_target?: Label [B, VD]
        result = {}
        if 'expression' in kwargs:
            if 'expression_tgt' in kwargs:
                expr_tgt = kwargs['expression_tgt']
            elif 'expr_ignore' in kwargs:
                expr_tgt = self.expression.ignore_labels(kwargs['expr_ignore'])
            else:
                expr_tgt = self.expression
            result.update(expr_tgt.smoothed_cross_entropy(kwargs['expression'], smoothing=0.01))

        if 'num_desc' in kwargs:
            num_loss = [gold.number_for_train.smoothed_cross_entropy(pred, smoothing=0.01)
                        for gold, pred in zip(self.description, kwargs['num_desc'])]
            result['num_desc'] = sum(num_loss) / len(num_loss)

        if 'var_desc' in kwargs:
            var_loss = [gold.variable_for_train.smoothed_cross_entropy(pred, smoothing=0.01)
                        for gold, pred in zip(self.description, kwargs['var_desc'])]
            result['var_desc'] = sum(var_loss) / len(var_loss)

        if 'var_len' in kwargs:
            var_len: torch.Tensor = kwargs['var_len_target']
            var_len_pred: torch.Tensor = kwargs['var_len']
            result['var_len'] = L1_LOSS(var_len_pred, var_len)

        return result


__all__ = ['Example', 'Text', 'Expression', 'ExpressionPrediction', 'Description',
           'ExtraInfo', 'Encoded', 'Label', 'Prediction']
