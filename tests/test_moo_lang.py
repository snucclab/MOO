from io import StringIO
from contextlib import redirect_stdout
from typing import Callable

import pytest
import random
import math
import itertools
import re
import sympy
from pathlib import Path
from common.solver.const import *

_ROOT_PATH = Path(__file__).parent.parent

_RESULT_NAME = [
    'result',
    'res',
    'answer',
    'r0',
    'value',
    'out'
]


def _verfity_indent(name: str, template: str):
    templates = template.split('\n')
    for lineno, line in enumerate(templates):
        if line.endswith('\n'):
            line = line[:-1]
        if line:
            assert re.fullmatch('^( {4})*[^\\s]+.*$', line), \
                '코드 들여쓰기는 반드시 공백문자 4개여야 합니다. (%s.pyt L#%3d)\n"%s"' % (name, lineno, line)


def _load_pyt(name: str):
    path = _ROOT_PATH / 'solver' / 'template' / (name + '.pyt')
    with path.open('r+t', encoding='UTF-8') as fp:
        lines = fp.readlines()

    lines = ''.join(lines)
    _verfity_indent(name, lines)
    return lines


def _exec_template(template: str, result: str, _locals=None, **kwargs):
    _global = {}
    _global['math'] = math
    _global['itertools'] = itertools

    _locals = _locals if _locals is not None else {}
    _code = template.format(**kwargs, result=result)
    print(_code)

    exec(_code, _global, _locals)
    return _locals.get(result, None)


def test_eq():
    template = _load_pyt(OPR_EQ)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_EQ)][CONVERT]

    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '5')) is True
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '7')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '"a"')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '"a"', '"a"')) is True
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5.0', '"a"')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5.0', '5.0')) is True
    assert _exec_template(template, _locals=dict(x=5), **converter(random.choice(_RESULT_NAME), 'x', '5.0')) is True
    assert _exec_template(template, _locals=dict(life=42),
                          **converter(random.choice(_RESULT_NAME), '42', 'life')) is True


def test_ADD():
    template = _load_pyt(OPR_ADD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_ADD)][CONVERT]

    for _ in range(10000):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x + y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x + y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_SUB():
    template = _load_pyt(OPR_SUB)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SUB)][CONVERT]

    for _ in range(10000):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x - y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x - y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_MUL():
    template = _load_pyt(OPR_MUL)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MUL)][CONVERT]

    for _ in range(10000):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x * y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x * y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_DIV():
    template = _load_pyt(OPR_DIV)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_DIV)][CONVERT]

    for _ in range(10000):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000
        if y == 0:
            y = 1

        assert (x / y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x / y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_MOD():
    template = _load_pyt(OPR_MOD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MOD)][CONVERT]

    for _ in range(10000):
        x = random.randint(10, 5000)
        y = random.randint(1, 50)

        assert (x % y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x % y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_POW():
    template = _load_pyt(OPR_POW)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_POW)][CONVERT]

    for _ in range(10000):
        x = random.random() * 20000
        y = random.random() * 5

        assert (x ** y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x ** y) == _exec_template(template, _locals=dict(x=x, y=y),
                                          **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_PRINT():
    template = _load_pyt(OPR_PRINT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_PRINT)][CONVERT]

    def _exec_out_catch(x):
        reader = StringIO()
        with redirect_stdout(reader):
            res = _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x)))
            assert res == None

        return reader.getvalue().strip()

    assert '정국' == _exec_out_catch('"정국"')
    assert '3' == _exec_out_catch(3)
    assert '3.05' == _exec_out_catch(3.05)
    assert '3.05' == _exec_out_catch(3.0492)
    assert '3.05' == _exec_out_catch(3.0512)
    assert '6.13' == _exec_out_catch(6.127)
    assert '6.13' == _exec_out_catch(6.132)
    assert '6.20' == _exec_out_catch(6.2)
    assert '6.20' == _exec_out_catch(6.201)
    assert '6.20' == _exec_out_catch(6.1999)


def test_SUM():
    template = _load_pyt(OPR_SUM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SUM)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert sum(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))


def test_LIST():
    template = _load_pyt(OPR_LIST)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST)][CONVERT]

    for _ in range(10000):
        result = _exec_template(template, **converter(random.choice(_RESULT_NAME)))
        assert type(result) is list
        assert len(result) == 0


def test_APPEND():
    template = _load_pyt(OPR_APPEND)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_APPEND)][CONVERT]

    for _ in range(10000):
        items = list()

        append = random.randint(1, 100)
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', str(append)))
        assert len(items) == 1
        assert items[-1] == append

        append = random.choice(_RESULT_NAME)
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', '"%s"' % append))
        assert len(items) == 2
        assert items[-1] == append

        append = random.random()
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', '"%s"' % append))
        assert len(items) == 3
        assert items[-1] == str(append)


def test_COMB():
    template = _load_pyt(OPR_COMB)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COMB)][CONVERT]

    for _ in range(10000):
        n = random.randint(0, 100)
        k = random.randint(0, n)

        result = math.comb(n, k)
        assert result == _exec_template(template, _locals=dict(n=n, k=k),
                                        **converter(random.choice(_RESULT_NAME), 'n', 'k'))
        assert result == _exec_template(template, _locals=dict(n=n),
                                        **converter(random.choice(_RESULT_NAME), 'n', k))
        assert result == _exec_template(template, _locals=dict(k=k),
                                        **converter(random.choice(_RESULT_NAME), n, 'k'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), n, k))


def test_PERM():
    template = _load_pyt(OPR_PERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_PERM)][CONVERT]

    for _ in range(10000):
        n = random.randint(0, 100)
        k = random.randint(0, n)

        result = math.perm(n, k)
        assert result == _exec_template(template, _locals=dict(n=n, k=k),
                                        **converter(random.choice(_RESULT_NAME), 'n', 'k'))
        assert result == _exec_template(template, _locals=dict(n=n),
                                        **converter(random.choice(_RESULT_NAME), 'n', k))
        assert result == _exec_template(template, _locals=dict(k=k),
                                        **converter(random.choice(_RESULT_NAME), n, 'k'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), n, k))


def test_MIN():
    template = _load_pyt(OPR_MIN)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MIN)][CONVERT]

    for _ in range(10000):
        length = random.randint(1, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert min(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [int(x) for x in items]
        assert min(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [str(x) for x in items]
        assert min(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [(x, random.random()) for x in items]
        min_value = min(items, key=lambda x: x[1])[0]
        assert min_value == _exec_template(template, _locals=dict(items=items),
                                           **converter(random.choice(_RESULT_NAME), 'items'))


def test_MAX():
    template = _load_pyt(OPR_MAX)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MAX)][CONVERT]

    for _ in range(10000):
        length = random.randint(1, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert max(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [int(x) for x in items]
        assert max(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [str(x) for x in items]
        assert max(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))

        items = [(x, random.random()) for x in items]
        max_value = max(items, key=lambda x: x[1])[0]
        assert max_value == _exec_template(template, _locals=dict(items=items),
                                           **converter(random.choice(_RESULT_NAME), 'items'))


def test_RANGE():
    template = _load_pyt(OPR_RANGE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_RANGE)][CONVERT]

    for _ in range(10000):
        start = random.randint(0, 200)
        end = random.randint(start, start + 200)
        step = random.randint(1, 5)

        result = list(range(start, end, step))
        assert result == _exec_template(template, _locals=dict(start=start, end=end, step=step),
                                        **converter(random.choice(_RESULT_NAME), 'start', 'end', 'step'))

        result = list(range(start, end, step))
        assert result == _exec_template(template, _locals=dict(end=end, step=step),
                                        **converter(random.choice(_RESULT_NAME), start, 'end', 'step'))

        result = list(range(start, end, step))
        assert result == _exec_template(template, _locals=dict(start=start, step=step),
                                        **converter(random.choice(_RESULT_NAME), 'start', end, 'step'))

        result = list(range(start, end, step))
        assert result == _exec_template(template, _locals=dict(start=start, end=end),
                                        **converter(random.choice(_RESULT_NAME), 'start', 'end', step))

        result = list(range(start, end, step))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), start, end, step))


def test_LCM():
    template = _load_pyt(OPR_LCM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LCM)][CONVERT]

    for _ in range(10000):
        length = random.randint(1, 10)
        items = [random.randint(1, 500) for _ in range(length)]
        lcm = int(sympy.lcm(items))

        assert lcm == _exec_template(template, _locals=dict(items=items),
                                     **converter(random.choice(_RESULT_NAME), 'items'))


def test_GCD():
    template = _load_pyt(OPR_GCD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GCD)][CONVERT]

    for _ in range(10000):
        length = random.randint(1, 10)
        items = [random.randint(1, 500) for _ in range(length)]
        gcd = int(sympy.gcd(items))

        assert gcd == _exec_template(template, _locals=dict(items=items),
                                     **converter(random.choice(_RESULT_NAME), 'items'))


def test_COUNT_MULTI():
    template = _load_pyt(OPR_COUNT_MULTI)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COUNT_MULTI)][CONVERT]

    for _ in range(10000):
        range_begin = random.randint(1, 1000)
        range_end = random.randint(range_begin + 100, range_begin + 10000)
        range_list = list(range(range_begin, range_end))
        multiples = [random.randint(2, 20)
                     for _ in range(random.randint(2, 6))]

        count_multiples = 0
        for i in range_list:
            if all(i % n == 0 for n in multiples):
                count_multiples += 1

        assert count_multiples == _exec_template(template, _locals=dict(ranges=range_list, multiples=multiples),
                                                 **converter(random.choice(_RESULT_NAME), 'ranges', 'multiples'))


def test_DIGIT():
    template = _load_pyt(OPR_DIGIT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_DIGIT)][CONVERT]

    for _ in range(10000):
        digit = random.randint(1, 10)
        unit = random.randint(1, 10)
        result = int(digit * 10 ** (unit - 1))

        assert result == _exec_template(template, _locals=dict(k=digit, n=unit),
                                        **converter(random.choice(_RESULT_NAME), 'k', 'n'))
        assert result == _exec_template(template, _locals=dict(n=unit),
                                        **converter(random.choice(_RESULT_NAME), digit, 'n'))
        assert result == _exec_template(template, _locals=dict(k=digit),
                                        **converter(random.choice(_RESULT_NAME), 'k', unit))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), digit, unit))


def test_TO_INT():
    template = _load_pyt(OPR_TO_INT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_TO_INT)][CONVERT]

    for _ in range(10000):
        number = random.random() * 100
        result = int(number)

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))


def test_REVERSE_DIGIT():
    template = _load_pyt(OPR_REVERSE_DIGIT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_REVERSE_DIGIT)][CONVERT]

    for _ in range(10000):
        number = random.randint(1, 100000)
        result = int(str(number)[::-1])

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))


def test_SEQ_TERM():
    template = _load_pyt(OPR_SEQ_TERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SEQ_TERM)][CONVERT]

    for _ in range(10000):
        diff = random.random() * 10
        base = random.random() * 100
        n = random.randint(50, 1000)
        samples = random.randint(3, 10)

        # 등차수열
        differences = [0] + [diff for _ in range(n - 1)]
        base_sequence = list(itertools.accumulate(differences, lambda x, y: x + y))
        diff_sequence = [base + x for x in base_sequence]

        last = diff_sequence[-1]
        diff_sequence = diff_sequence[:samples]
        assert last == _exec_template(template, _locals=dict(samples=diff_sequence, n=n),
                                      **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
        assert last == _exec_template(template, _locals=dict(samples=diff_sequence),
                                      **converter(random.choice(_RESULT_NAME), 'samples', n))

        # 등차 빈칸
        if samples > 5:
            n_replace = random.randint(0, samples)
            sequence = diff_sequence.copy()
            answer = sequence[n_replace]
            sequence[n_replace] = 'A'
            assert answer == _exec_template(template, _locals=dict(samples=sequence, n=n_replace + 1),
                                            **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
            assert answer == _exec_template(template, _locals=dict(samples=sequence),
                                            **converter(random.choice(_RESULT_NAME), 'samples', n_replace + 1))

        # 계차수열
        diff_sequence = [0] + diff_sequence[:-1]
        acc_seqeunce = list(itertools.accumulate(diff_sequence, lambda x, y: x + y))
        base = random.random() * 100
        sequence = [base + x for x in acc_seqeunce]
        assert sequence[-1] == _exec_template(template, _locals=dict(samples=sequence[:samples], n=n),
                                              **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
        assert sequence[-1] == _exec_template(template, _locals=dict(samples=sequence[:samples]),
                                              **converter(random.choice(_RESULT_NAME), 'samples', n))

        # 계차 빈칸
        if samples > 5:
            n_replace = random.randint(0, samples)
            answer = sequence[n_replace]
            sequence[n_replace] = 'A'
            assert answer == _exec_template(template, _locals=dict(samples=sequence, n=n_replace + 1),
                                            **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
            assert answer == _exec_template(template, _locals=dict(samples=sequence),
                                            **converter(random.choice(_RESULT_NAME), 'samples', n_replace + 1))


def test_REP_SEQ_TERM():
    template = _load_pyt(OPR_REP_SEQ_TERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_REP_SEQ_TERM)][CONVERT]

    for _ in range(10000):
        n = random.randint(50, 1000)
        samples = random.randint(3, 10)
        sample_seq = [random.random() * 100 for _ in range(samples)]
        sample_seq += sample_seq

        # 반복 수열
        answer = sample_seq[(n - 1) % len(sample_seq)]
        assert answer == _exec_template(template, _locals=dict(samples=sample_seq, n=n),
                                        **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
        assert answer == _exec_template(template, _locals=dict(samples=sample_seq),
                                        **converter(random.choice(_RESULT_NAME), 'samples', n))

        if samples > 5:
            # 반복 빈칸
            n_replace = random.randint(0, samples * 2)
            answer = sample_seq[n_replace]
            sample_seq[n_replace] = 'A'
            assert answer == _exec_template(template, _locals=dict(samples=sample_seq, n=n_replace + 1),
                                            **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
            assert answer == _exec_template(template, _locals=dict(samples=sample_seq),
                                            **converter(random.choice(_RESULT_NAME), 'samples', n_replace + 1))
            assert answer == _exec_template(template, _locals=dict(samples=sample_seq, n=n),
                                            **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
            assert answer == _exec_template(template, _locals=dict(samples=sample_seq),
                                            **converter(random.choice(_RESULT_NAME), 'samples', n))


def test_MAKE_PAIR():
    template = _load_pyt(OPR_MAKE_PAIR)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MAKE_PAIR)][CONVERT]

    for _ in range(10000):
        name = random.choice(['상규', '지원', '윤재', '지수', '해룡'])
        number = random.randint(1, 100000)

        result = (name, number)
        assert result == _exec_template(template, _locals=dict(name=name, value=number),
                                        **converter(random.choice(_RESULT_NAME), 'name', 'value'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), '"%s"' % name, number))

        result = (name, name)
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), '"%s"' % name, '"%s"' % name))


def test_COUNT():
    template = _load_pyt(OPR_COUNT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COUNT)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert len(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))


def test_LT():
    template = _load_pyt(OPR_LT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LT)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x < base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_LE():
    template = _load_pyt(OPR_LE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LE)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x <= base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_GT():
    template = _load_pyt(OPR_GT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GT)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x > base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_GE():
    template = _load_pyt(OPR_GE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GE)][CONVERT]

    for _ in range(10000):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x >= base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_LIST_CONCAT():
    template = _load_pyt(OPR_LIST_CONCAT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_CONCAT)][CONVERT]

    for _ in range(10000):
        items1 = [random.random() * 100 for _ in range(random.randint(0, 200))]
        items2 = [random.random() * 100 for _ in range(random.randint(0, 200))]

        result = items1 + items2
        assert result == _exec_template(template, _locals=dict(item1=items1, item2=items2),
                                        **converter(random.choice(_RESULT_NAME), 'item1', 'item2'))


def test_LIST_INDEX():
    template = _load_pyt(OPR_LIST_INDEX)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_INDEX)][CONVERT]

    for _ in range(10000):
        items = [random.random() * 100 for _ in range(random.randint(1, 200))]
        item = random.choice(items)
        index = items.index(item)

        assert index == _exec_template(template, _locals=dict(items=items, item=item),
                                       **converter(random.choice(_RESULT_NAME), 'items', 'item'))
        assert index == _exec_template(template, _locals=dict(items=items),
                                       **converter(random.choice(_RESULT_NAME), 'items', item))

        items = [str(x) for x in items]
        assert index == _exec_template(template, _locals=dict(items=items, item=str(item)),
                                       **converter(random.choice(_RESULT_NAME), 'items', 'item'))
        assert index == _exec_template(template, _locals=dict(items=items),
                                       **converter(random.choice(_RESULT_NAME), 'items', '"%s"' % item))


def test_LIST_REPLACE():
    template = _load_pyt(OPR_LIST_REPLACE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_REPLACE)][CONVERT]

    for _ in range(10000):
        items = [random.random() * 100 for _ in range(random.randint(1, 200))]
        index = random.randint(0, len(items) - 1)
        replacement = 'ABC'

        replaced = items.copy()
        replaced[index] = replacement

        assert replaced == _exec_template(template, _locals=dict(items=items, index=index, replace=replacement),
                                          **converter(random.choice(_RESULT_NAME), 'items', 'index', 'replace'))
        assert replaced == _exec_template(template, _locals=dict(items=items, replace=replacement),
                                          **converter(random.choice(_RESULT_NAME), 'items', index, 'replace'))
        assert replaced == _exec_template(template, _locals=dict(items=items, index=index),
                                          **converter(random.choice(_RESULT_NAME), 'items', 'index',
                                                      '"%s"' % replacement))
        assert replaced == _exec_template(template, _locals=dict(items=items),
                                          **converter(random.choice(_RESULT_NAME), 'items', index,
                                                      '"%s"' % replacement))


def test_CEIL():
    template = _load_pyt(OPR_CEIL)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_CEIL)][CONVERT]

    for _ in range(10000):
        number = random.random() * 100
        result = int(math.ceil(number))

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))
