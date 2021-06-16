import itertools
import math
import random
import re
from contextlib import redirect_stdout
from io import StringIO

import sympy

from common.solver.const import *
from common.sys.const import MODULE_RSC_PATH

_RESULT_NAME = [
    'result',
    'res',
    'answer',
    'r0',
    'value',
    'out'
]


def _verify_indent(name: str, template: str):
    templates = template.split('\n')
    for lineno, line in enumerate(templates):
        if line.endswith('\n'):
            line = line[:-1]
        if line:
            assert re.fullmatch('^( {4})*[^\\s]+.*$', line), \
                '코드 들여쓰기는 반드시 공백문자 4개여야 합니다. (%s.pyt L#%3d)\n\'%s\'' % (name, lineno, line)
            assert '"' not in line, \
                '코드에는 쌍따옴표를 사용할 수 없습니다. (%s.pyt L#%3d)\n"%s"' % (name, lineno, line)


def _load_pyt(name: str):
    path = MODULE_RSC_PATH / 'moo_lang' / (name + '.pyt')
    with path.open('r+t', encoding='UTF-8') as fp:
        lines = fp.readlines()

    lines = ''.join(lines)
    _verify_indent(name, lines)
    return lines


def _exec_template(template: str, result: str, _locals=None, **kwargs):
    _global = {}
    _global['math'] = math
    _global['itertools'] = itertools
    if '##@@@@' in template:
        _global['sympy'] = sympy
        _global['re'] = re

    _locals = _locals if _locals is not None else {}
    _code = template.format(**kwargs, result=result)

    exec(_code, _global, _locals)
    return _locals.get(result, None)


def test_eq():
    template = _load_pyt(OPR_EQ)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_EQ)][CONVERT]

    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '5')) is True
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '7')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5', '\'a\'')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '\'a\'', '\'a\'')) is True
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5.0', '\'a\'')) is False
    assert _exec_template(template, **converter(random.choice(_RESULT_NAME), '5.0', '5.0')) is True
    assert _exec_template(template, _locals=dict(x=5), **converter(random.choice(_RESULT_NAME), 'x', '5.0')) is True
    assert _exec_template(template, _locals=dict(life=42),
                          **converter(random.choice(_RESULT_NAME), '42', 'life')) is True


def test_add():
    template = _load_pyt(OPR_ADD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_ADD)][CONVERT]

    for _ in range(500):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x + y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x + y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_sub():
    template = _load_pyt(OPR_SUB)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SUB)][CONVERT]

    for _ in range(500):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x - y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x - y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_mul():
    template = _load_pyt(OPR_MUL)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MUL)][CONVERT]

    for _ in range(500):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000

        assert (x * y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x * y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_div():
    template = _load_pyt(OPR_DIV)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_DIV)][CONVERT]

    for _ in range(500):
        x = random.random() * 20000 - 10000
        y = random.random() * 20000 - 10000
        if y == 0:
            y = 1

        assert (x / y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x / y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_mod():
    template = _load_pyt(OPR_MOD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MOD)][CONVERT]

    for _ in range(500):
        x = random.randint(10, 5000)
        y = random.randint(1, 50)

        assert (x % y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x % y) == _exec_template(template, _locals=dict(x=x, y=y),
                                         **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_pow():
    template = _load_pyt(OPR_POW)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_POW)][CONVERT]

    for _ in range(500):
        x = random.random() * 20000
        y = random.random() * 5

        assert (x ** y) == _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x), str(y)))
        assert (x ** y) == _exec_template(template, _locals=dict(x=x, y=y),
                                          **converter(random.choice(_RESULT_NAME), 'x', 'y'))


def test_print():
    template = _load_pyt(OPR_PRINT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_PRINT)][CONVERT]

    def _exec_out_catch(x):
        reader = StringIO()
        with redirect_stdout(reader):
            res = _exec_template(template, **converter(random.choice(_RESULT_NAME), str(x)))
            assert res == None

        return reader.getvalue().strip()

    assert '정국' == _exec_out_catch('\'정국\'')
    assert '3' == _exec_out_catch(3)
    assert '3.05' == _exec_out_catch(3.05)
    assert '3.05' == _exec_out_catch(3.0492)
    assert '3.05' == _exec_out_catch(3.0512)
    assert '6.13' == _exec_out_catch(6.127)
    assert '6.13' == _exec_out_catch(6.132)
    assert '6.20' == _exec_out_catch(6.2)
    assert '6.20' == _exec_out_catch(6.201)
    assert '6.20' == _exec_out_catch(6.1999)


def test_sum():
    template = _load_pyt(OPR_SUM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SUM)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert sum(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))


def test_list():
    template = _load_pyt(OPR_LIST)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST)][CONVERT]

    for _ in range(500):
        result = _exec_template(template, **converter(random.choice(_RESULT_NAME)))
        assert type(result) is list
        assert len(result) == 0


def test_append():
    template = _load_pyt(OPR_APPEND)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_APPEND)][CONVERT]

    for _ in range(500):
        items = list()

        append = random.randint(1, 100)
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', str(append)))
        assert len(items) == 1
        assert items[-1] == append

        append = random.choice(_RESULT_NAME)
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', '\'%s\'' % append))
        assert len(items) == 2
        assert items[-1] == append

        append = random.random()
        items = _exec_template(template, _locals=dict(items=items),
                               **converter(random.choice(_RESULT_NAME), 'items', '\'%s\'' % append))
        assert len(items) == 3
        assert items[-1] == str(append)


def test_comb():
    template = _load_pyt(OPR_COMB)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COMB)][CONVERT]

    for _ in range(500):
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


def test_perm():
    template = _load_pyt(OPR_PERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_PERM)][CONVERT]

    for _ in range(500):
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


def test_min():
    template = _load_pyt(OPR_MIN)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MIN)][CONVERT]

    for _ in range(500):
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


def test_max():
    template = _load_pyt(OPR_MAX)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MAX)][CONVERT]

    for _ in range(500):
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


def test_range():
    template = _load_pyt(OPR_RANGE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_RANGE)][CONVERT]

    for _ in range(500):
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


def test_lcm():
    template = _load_pyt(OPR_LCM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LCM)][CONVERT]

    for _ in range(500):
        length = random.randint(1, 10)
        items = [random.randint(1, 500) for _ in range(length)]
        lcm = int(sympy.lcm(items))

        assert lcm == _exec_template(template, _locals=dict(items=items),
                                     **converter(random.choice(_RESULT_NAME), 'items'))


def test_gcd():
    template = _load_pyt(OPR_GCD)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GCD)][CONVERT]

    for _ in range(500):
        length = random.randint(1, 10)
        items = [random.randint(1, 500) for _ in range(length)]
        gcd = int(sympy.gcd(items))

        assert gcd == _exec_template(template, _locals=dict(items=items),
                                     **converter(random.choice(_RESULT_NAME), 'items'))


def test_count_multi():
    template = _load_pyt(OPR_COUNT_MULTI)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COUNT_MULTI)][CONVERT]

    for _ in range(500):
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


def test_digit():
    template = _load_pyt(OPR_DIGIT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_DIGIT)][CONVERT]

    for _ in range(500):
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


def test_to_int():
    template = _load_pyt(OPR_TO_INT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_TO_INT)][CONVERT]

    for _ in range(500):
        number = random.random() * 100
        result = int(number)

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))


def test_reverse_digit():
    template = _load_pyt(OPR_REVERSE_DIGIT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_REVERSE_DIGIT)][CONVERT]

    for _ in range(500):
        number = random.randint(1, 100000)
        result = int(str(number)[::-1])

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))


def test_seq_term():
    template = _load_pyt(OPR_SEQ_TERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_SEQ_TERM)][CONVERT]
    symbol = sympy.Symbol('x')

    for _ in range(500):
        degree = random.randint(2, 5)
        polynomial = sympy.polys.specialpolys.random_poly(symbol, degree, -10, 10)
        samples = random.randint(degree + 2, 10)
        n = random.randint(20, 100)

        sequence = [int(polynomial.subs({symbol: i + 1})) for i in range(samples)]
        answer = int(polynomial.subs({symbol: n}))

        # 빈칸 없을 때
        assert abs(answer - _exec_template(template, _locals=dict(samples=sequence, n=n),
                                           **converter(random.choice(_RESULT_NAME), 'samples', 'n'))) < 1E-5
        assert abs(answer - _exec_template(template, _locals=dict(samples=sequence),
                                           **converter(random.choice(_RESULT_NAME), 'samples', n))) < 1E-5

        # 빈칸 있을 때
        n = random.randint(1, samples)
        answer = sequence[n - 1]
        sequence[n - 1] = 'A'
        assert abs(answer - _exec_template(template, _locals=dict(samples=sequence, n=n),
                                           **converter(random.choice(_RESULT_NAME), 'samples', 'n'))) < 1E-5
        assert abs(answer - _exec_template(template, _locals=dict(samples=sequence),
                                           **converter(random.choice(_RESULT_NAME), 'samples', n))) < 1E-5


def test_rep_seq_term():
    template = _load_pyt(OPR_REP_SEQ_TERM)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_REP_SEQ_TERM)][CONVERT]

    for _ in range(500):
        n = random.randint(50, 1000)
        samples = random.randint(3, 10)
        sample_seq = [random.randint(0, 10000) / 100 for _ in range(samples)]
        sample_seq += sample_seq

        # 반복 수열
        answer = sample_seq[(n - 1) % len(sample_seq)]
        assert abs(answer - _exec_template(template, _locals=dict(samples=sample_seq, n=n),
                                           **converter(random.choice(_RESULT_NAME), 'samples', 'n'))) < 1E-5
        assert abs(answer - _exec_template(template, _locals=dict(samples=sample_seq),
                                           **converter(random.choice(_RESULT_NAME), 'samples', n))) < 1E-5

        if samples > 5:
            # 반복 빈칸
            n_replace = random.randint(0, samples * 2 - 1)
            answer_replace = sample_seq[n_replace]
            sample_seq[n_replace] = 'A'
            assert abs(answer_replace - _exec_template(template, _locals=dict(samples=sample_seq, n=n_replace + 1),
                                                       **converter(random.choice(_RESULT_NAME), 'samples', 'n'))) < 1E-5
            assert abs(answer_replace - _exec_template(template, _locals=dict(samples=sample_seq),
                                                       **converter(random.choice(_RESULT_NAME), 'samples',
                                                                   n_replace + 1))) < 1E-5
            assert abs(answer - _exec_template(template, _locals=dict(samples=sample_seq, n=n),
                                               **converter(random.choice(_RESULT_NAME), 'samples', 'n'))) < 1E-5
            assert abs(answer - _exec_template(template, _locals=dict(samples=sample_seq),
                                               **converter(random.choice(_RESULT_NAME), 'samples', n))) < 1E-5

    boxes = list('가나다라마바사아자차카타파하')
    for _ in range(500):
        sample_seq = random.sample(boxes, random.randint(4, 10))
        n = random.randint(50, 1000)

        # 반복 수열
        answer = sample_seq[(n - 1) % len(sample_seq)]
        assert answer == _exec_template(template, _locals=dict(samples=sample_seq, n=n),
                                        **converter(random.choice(_RESULT_NAME), 'samples', 'n'))
        assert answer == _exec_template(template, _locals=dict(samples=sample_seq),
                                        **converter(random.choice(_RESULT_NAME), 'samples', n))


def test_make_pair():
    template = _load_pyt(OPR_MAKE_PAIR)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_MAKE_PAIR)][CONVERT]

    for _ in range(500):
        name = random.choice(['상규', '지원', '윤재', '지수', '해룡'])
        number = random.randint(1, 100000)

        result = (name, number)
        assert result == _exec_template(template, _locals=dict(name=name, value=number),
                                        **converter(random.choice(_RESULT_NAME), 'name', 'value'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), '\'%s\'' % name, number))

        result = (name, name)
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), '\'%s\'' % name, '\'%s\'' % name))


def test_count():
    template = _load_pyt(OPR_COUNT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_COUNT)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]

        assert len(items) == _exec_template(template, _locals=dict(items=items),
                                            **converter(random.choice(_RESULT_NAME), 'items'))


def test_lt():
    template = _load_pyt(OPR_LT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LT)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x < base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_le():
    template = _load_pyt(OPR_LE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LE)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x <= base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_gt():
    template = _load_pyt(OPR_GT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GT)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x > base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_ge():
    template = _load_pyt(OPR_GE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_GE)][CONVERT]

    for _ in range(500):
        length = random.randint(0, 200)
        items = [random.random() * 100 for _ in range(length)]
        base_value = random.random() * 100

        diff = [int(x >= base_value) for x in items]
        assert diff == _exec_template(template, _locals=dict(items=items),
                                      **converter(random.choice(_RESULT_NAME), 'items', base_value))
        assert diff == _exec_template(template, _locals=dict(items=items, base=base_value),
                                      **converter(random.choice(_RESULT_NAME), 'items', 'base'))


def test_list_concat():
    template = _load_pyt(OPR_LIST_CONCAT)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_CONCAT)][CONVERT]

    for _ in range(500):
        items1 = [random.random() * 100 for _ in range(random.randint(0, 200))]
        items2 = [random.random() * 100 for _ in range(random.randint(0, 200))]

        result = items1 + items2
        assert result == _exec_template(template, _locals=dict(item1=items1, item2=items2),
                                        **converter(random.choice(_RESULT_NAME), 'item1', 'item2'))


def test_list_index():
    template = _load_pyt(OPR_LIST_INDEX)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_INDEX)][CONVERT]

    for _ in range(500):
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
                                       **converter(random.choice(_RESULT_NAME), 'items', '\'%s\'' % item))


def test_list_replace():
    template = _load_pyt(OPR_LIST_REPLACE)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_REPLACE)][CONVERT]

    for _ in range(500):
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
                                                      '\'%s\'' % replacement))
        assert replaced == _exec_template(template, _locals=dict(items=items),
                                          **converter(random.choice(_RESULT_NAME), 'items', index,
                                                      '\'%s\'' % replacement))


def test_ceil():
    template = _load_pyt(OPR_CEIL)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_CEIL)][CONVERT]

    for _ in range(500):
        number = random.random() * 100
        result = int(math.ceil(number))

        assert result == _exec_template(template, _locals=dict(n=number),
                                        **converter(random.choice(_RESULT_NAME), 'n'))
        assert result == _exec_template(template,
                                        **converter(random.choice(_RESULT_NAME), number))


def test_list_mul():
    template = _load_pyt(OPR_LIST_MUL)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_LIST_MUL)][CONVERT]

    for _ in range(500):
        items = [random.random() * 100 for _ in range(random.randint(0, 200))]
        multiple = random.randint(1, 10)

        result = items * multiple
        assert result == _exec_template(template, _locals=dict(items=items, n=multiple),
                                        **converter(random.choice(_RESULT_NAME), 'items', 'n'))
        assert result == _exec_template(template, _locals=dict(items=items),
                                        **converter(random.choice(_RESULT_NAME), 'items', multiple))


def test_call_sympy():
    template = _load_pyt(OPR_CALL_SYMPY)
    converter = OPR_VALUES[OPR_TOKENS.index(OPR_CALL_SYMPY)][CONVERT]

    # Multi digit sum/subtraction
    for _ in range(500):
        digits = random.randint(2, 5)
        a = random.randint(10 ** (digits - 1), 10 ** digits - 1)
        b = random.randint(10 ** (digits - 1), 10 ** digits - 1)
        abc = [list(str(a)), list(str(b)), list(str(a + b))]

        expected = {}
        for d in range(1, digits + 1):
            if random.random() < 0.2:
                continue

            character = chr(ord('A') + d - 1)
            choice = random.randint(0, 2)

            expected[character] = abc[choice][-d]
            abc[choice][-d] = character

        if random.random() < 0.5:
            # addition
            equation = '%s+%s=%s' % (''.join(abc[0]), ''.join(abc[1]), ''.join(abc[2]))
        else:
            # subtraction
            equation = '%s-%s=%s' % (''.join(abc[2]), ''.join(abc[0]), ''.join(abc[1]))

        if len(expected.keys()) == 0:
            continue

        target = random.choice(list(expected.keys()))
        answer = int(expected[target])
        assert answer == _exec_template(template, _locals=dict(equation=[equation], target=target),
                                        **converter(random.choice(_RESULT_NAME), 'equation', 'target'))
        assert answer == _exec_template(template, _locals=dict(equation=[equation]),
                                        **converter(random.choice(_RESULT_NAME), 'equation', '\'%s\'' % target))

    # System of equation
    for _ in range(500):
        arguments = random.randint(2, 4)
        variables = [random.randint(0, 100) / 4 for _ in range(arguments)]
        keys = [chr(ord('U') + d - 1) for d in range(arguments)]
        equations = [[random.randint(-100, 100) / 4 for _ in range(arguments)]
                     for _ in range(arguments)]

        eq_built = []
        for eq in equations:
            result = sum([a * x for a, x in zip(eq, variables)])
            equation = ''.join(['+%s*%s' % (a, x) if a > 0 else ('%s*%s' % (a, x) if a < 0 else '')
                                for a, x in zip(eq, keys)]) + '=' + str(result)
            eq_built.append(equation)

        expected = {key: value for key, value in zip(keys, variables)}
        # 빈칸 없을 때
        target = random.choice(keys)
        answer = expected[target]
        assert answer == _exec_template(template, _locals=dict(equation=eq_built, target=target),
                                        **converter(random.choice(_RESULT_NAME), 'equation', 'target'))
        assert answer == _exec_template(template, _locals=dict(equation=eq_built),
                                        **converter(random.choice(_RESULT_NAME), 'equation', '\'%s\'' % target))
