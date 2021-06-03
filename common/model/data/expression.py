import logging
from typing import List, Dict, Optional, Tuple

import torch

from common.const.operand import *
from common.const.operator import *
from common.const.pad import PAD_ID
from common.seed.parse import parse_infix, RELATION_CLASSES
from common.model.base import TypeTensorBatchable, TypeSelectable
from .label import Label
from .prediction import Prediction


def _operator_reader(token: int) -> str:
    return '' if token == PAD_ID else OPR_TOKENS[token]


def _operand_reader(token: int) -> str:
    if token == PAD_ID:
        return ''
    elif token < CON_END:
        return CON_TOKENS[token]
    elif token < NUM_END:
        return NUM_FORMAT % (token - NUM_BEGIN)
    else:
        return RES_FORMAT % (token - RES_BEGIN)


def _operand_to_sympy(token: int, workspace: list) -> Optional[sympy.Expr]:
    if token == PAD_ID:
        return None
    elif token < CON_END:
        return sympy.Number(CON_TOKENS[token])
    elif token < NUM_END:
        return sympy.Symbol(NUM_FORMAT % (token - NUM_BEGIN), real=True)
    else:
        rid = token - RES_BEGIN
        return workspace[rid] if rid < len(workspace) else None


def _read_expression(operator: list, operands: list, shape: tuple) -> dict:
    result = []

    if len(shape) == 1:
        operator = [operator]
        operands = [[operand_j] for operand_j in operands]

    # Expression: [B, T] and A * [B, T]
    for b in range(len(operator)):
        func = operator[b]
        args = list(zip(*[operand_j[b] for operand_j in operands]))
        expr_b = []

        counter = 0
        for t, f_t in enumerate(func):
            if f_t == 0:
                counter = 0
            elif f_t == -1:
                break
            else:
                expr_b.append('%s: %s(%s)' % (RES_FORMAT % counter, _operator_reader(f_t),
                                              ', '.join([_operand_reader(a) for a in args[t]])))
                counter += 1

        result.append(' '.join(expr_b))

    if len(result) == 1:
        return dict(shape=list(shape), tokens=result[0])
    else:
        return dict(shape=list(shape), tokens=result)


class ExpressionPrediction(TypeSelectable):
    #: Operator prediction
    operator: Prediction
    #: List of operand predictions, [operand0, operand1, ...]. Each item is either batched or not.
    operands: List[Prediction]

    def __init__(self, operator: Prediction, operands: List[Prediction]):
        super().__init__()
        self.operator = operator
        self.operands = operands

    @classmethod
    def from_tensors(cls, operator: torch.Tensor, operands: List[torch.Tensor]) -> 'ExpressionPrediction':
        return ExpressionPrediction(operator=Prediction(operator),
                                    operands=[Prediction(operand_j) for operand_j in operands])

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)

    def to_human_readable(self) -> dict:
        operator = self.operator.topmost_index.tolist()
        operands = [operand_j.topmost_index.tolist()
                    for operand_j in self.operands]

        return _read_expression(operator, operands, self.operator.shape)


class Expression(TypeTensorBatchable, TypeSelectable):
    #: Operator label
    operator: Label
    #: List of operand labels, [operand0, operand1, ...]. Each item is either batched or not.
    operands: List[Label]

    def __init__(self, operator: Label, operands: List[Label]):
        super().__init__()
        assert all(operator.shape == operand_j.shape for operand_j in operands)
        assert len(operands) == OPR_MAX_ARITY
        self.operator = operator
        self.operands = operands

    @property
    def shape(self) -> torch.Size:
        return self.operator.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.operator.pad

    @property
    def device(self) -> torch.device:
        return self.operator.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.operator.sequence_lengths

    @classmethod
    def from_dict(cls, raw: dict, var_list_out: List[str], field: str) -> 'Expression':
        expressions = []
        equations = raw[field]

        # Sort equation by number occurrences
        equations = sorted(equations, key=lambda eq: tuple(sorted(eq['used']['numbers'])))

        # Collect all variables in the order of occurrence
        for eqn in equations:
            for tok in eqn['formula'].split():
                if tok.startswith(NUM_PREFIX) or not tok[0].isalpha():
                    continue
                if tok not in var_list_out:
                    var_list_out.append(tok)
                    expressions.append((OPR_NEW_VAR_ID,) + (PAD_ID,) * OPR_MAX_ARITY)

        # Parse expression
        for eqn in equations:
            expressions += parse_infix(eqn['formula'], var_list_out, offset=len(expressions))

        # Prepend NEW_EQN() and Append DONE()
        # Note: NEW_EQN() is inserted here because it should be ignored when numbering results.
        expressions.insert(0, (OPR_NEW_EQN_ID,) + (PAD_ID,) * OPR_MAX_ARITY)
        expressions.append((OPR_DONE_ID,) + (PAD_ID,) * OPR_MAX_ARITY)

        # Separate operator and operands
        operator, *operands = zip(*expressions)

        return Expression(operator=Label.from_list(operator),
                          operands=[Label.from_list(operand_j) for operand_j in operands])

    @classmethod
    def build_batch(cls, *items: 'Expression') -> 'Expression':
        operands = zip(*[item.operands for item in items])
        return Expression(operator=Label.build_batch(*[item.operator for item in items]),
                          operands=[Label.build_batch(*operand_j) for operand_j in operands])

    @classmethod
    def concat(cls, *items: 'Expression', dim: int = 0) -> 'Expression':
        operands = zip(*[item.operands for item in items])
        return Expression(operator=Label.concat(*[item.operator for item in items], dim=dim),
                          operands=[Label.concat(*operand_j, dim=dim) for operand_j in operands])

    @classmethod
    def from_tensors(cls, operator: torch.LongTensor, operands: List[torch.LongTensor]) -> 'Expression':
        return Expression(operator=Label(operator),
                          operands=[Label(operand_j) for operand_j in operands])

    @classmethod
    def get_generation_base(cls) -> 'Expression':
        operator = Label(torch.LongTensor([[OPR_NEW_EQN_ID]]))  # [M=1, T=1]
        operands = [Label(torch.LongTensor([[PAD_ID]]))  # [M=1, T=1]
                    for _ in range(OPR_MAX_ARITY)]
        return Expression(operator, operands)

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)

    def smoothed_cross_entropy(self, pred: ExpressionPrediction, smoothing: float = 0.1, focal: float = 0.0,
                               shift_target: bool = True) -> Dict[str, torch.Tensor]:
        assert len(pred.operands) == len(self.operands)
        return dict(
            operator=self.operator.smoothed_cross_entropy(pred.operator, smoothing, focal, shift_target),
            **{'operand%s' % j: gold_j.smoothed_cross_entropy(pred_j, smoothing, focal, shift_target)
               for j, (gold_j, pred_j) in enumerate(zip(self.operands, pred.operands))}
        )

    def accuracy_of(self, pred: ExpressionPrediction) -> Dict[str, float]:
        assert len(pred.operands) == len(self.operands)

        operator_acc = self.operator.accuracy_of(pred.operator)
        operands_acc = [gold_j.accuracy_of(pred_j)
                        for gold_j, pred_j in zip(self.operands, pred.operands)]

        all_seq = torch.stack([operator_acc.pop('seq')] + [acc_j.pop('seq') for acc_j in operands_acc]).prod(dim=0)
        all_seq = float(all_seq.prod(dim=1).float().mean())

        return dict(
            seq_acc_all=all_seq,
            **{key + '_operator': value
               for key, value in operator_acc.items()},
            **{key + '_operand%s' % j: value
               for j, acc_j in enumerate(operands_acc)
               for key, value in acc_j.items()}
        )

    def extends_to(self, next_operator: torch.LongTensor, next_operands: List[torch.LongTensor]) -> 'Expression':
        return Expression(operator=self.operator.extends_to(next_operator),
                          operands=[prev_j.extends_to(next_j)
                                    for prev_j, next_j in zip(self.operands, next_operands)])

    def unsqueeze(self, dim: int) -> 'Expression':
        return Expression(operator=self.operator.unsqueeze(dim),
                          operands=[operand_j.unsqueeze(dim)
                                    for operand_j in self.operands])

    def to_human_readable(self, **kwargs) -> dict:
        operator = self.operator.indices.tolist()
        operands = [operand_j.indices.tolist()
                    for operand_j in self.operands]

        return _read_expression(operator, operands, self.operator.shape)

    def to_sympy(self, var_list: List[str]) -> List[sympy.Expr]:
        assert not self.is_batched

        func: List[int] = self.operator.indices.tolist()
        args: List[Tuple[int, ...]] = list(zip(*[operand_j.indices.tolist() for operand_j in self.operands]))

        workspace = []
        var_index = 0

        try:
            for f, a in zip(func, args):
                if f == OPR_NEW_EQN_ID:
                    workspace.clear()
                    continue
                if f == OPR_DONE_ID:
                    break

                if f == OPR_NEW_VAR_ID:
                    if len(var_list) > var_index:
                        var_name = var_list[var_index]
                    else:
                        var_name = VAR_FORMAT % var_index

                    workspace.append(sympy.Symbol(var_name, real=True))
                    var_index += 1
                else:
                    info = OPR_VALUES[f]
                    operator = info[KEY_CONVERT]

                    arity = info[KEY_ARITY]
                    operands = [_operand_to_sympy(a_j, workspace) for a_j in a[:arity]]
                    if len(operands) < arity:
                        missed = arity - len(operands)
                        logging.warning('Formula has %s missing argument(s): %s%s' % (missed, f, repr(tuple(a))))
                        # Append '0' for empty spaces
                        operands += [sympy.Number(0)] * missed

                    workspace.append(operator(*operands))
        except Exception as e:
            logging.warning('We ignored the following issue on converting expressions, and returned [].', exc_info=e)
            return []

        return [expr for expr in workspace if isinstance(expr, RELATION_CLASSES)]

    def ignore_labels(self, expr_excluded: set) -> 'Expression':
        return Expression(operator=self.operator.ignore_labels(expr_excluded),
                          operands=self.operands)

    def treat_variables_as_numbers(self, number_max: list) -> 'Expression':
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        new_batch = []
        for b in range(batch_sz):
            operator = self.operator.indices[b].tolist()  # [T]
            operands = [operand[b].indices.tolist() for operand in self.operands]  # [T]

            new_operator = []
            new_operands = [[] for _ in operands]
            var_map = {}
            for t in range(len(operator)):
                if operator[t] == OPR_NEW_VAR_ID:
                    var_map[RES_BEGIN + len(var_map)] = NUM_BEGIN + number_max[b] + len(var_map)
                    # Do not append NEW_VAR here
                    continue

                new_operator.append(operator[t])
                for j in range(OPR_MAX_ARITY):
                    op_j = operands[j][t]
                    if op_j in var_map:
                        new_operands[j].append(var_map[op_j])
                    elif op_j >= RES_BEGIN:
                        new_operands[j].append(op_j - len(var_map))
                    else:
                        new_operands[j].append(op_j)

            new_batch.append(Expression(Label.from_list(new_operator),
                                        [Label.from_list(operand_j) for operand_j in new_operands]))

        return Expression.build_batch(*new_batch).to(self.device)

    def restore_variables(self, number_max: list, variable_max: list) -> 'Expression':
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        new_batch = []
        for b in range(batch_sz):
            operator = self.operator.indices[b].tolist()  # [T]
            operands = [operand[b].indices.tolist() for operand in self.operands]  # [T]

            num_end = NUM_BEGIN + number_max[b]
            var_len = variable_max[b]

            new_operator = []
            new_operands = [[] for _ in operands]
            for t in range(len(operator)):
                new_operator.append(operator[t])
                for j in range(OPR_MAX_ARITY):
                    op_j = operands[j][t]
                    if num_end <= op_j < RES_BEGIN:
                        new_operands[j].append(op_j - num_end + RES_BEGIN)
                    elif op_j >= RES_BEGIN:
                        new_operands[j].append(op_j + var_len)
                    else:
                        new_operands[j].append(op_j)

                if operator[t] == OPR_NEW_EQN_ID:
                    # Append variables
                    new_operator += [OPR_NEW_VAR_ID] * var_len
                    for new_op in new_operands:
                        new_op += [PAD_ID] * var_len

            new_batch.append(Expression(Label.from_list(new_operator),
                                        [Label.from_list(operand_j) for operand_j in new_operands]))

        return Expression.build_batch(*new_batch).to(self.device)
