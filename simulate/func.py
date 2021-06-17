import math
import random
from decimal import Decimal

import sympy

UNK = [chr(ord('A') + i) for i in range(26)]


def arithmetic_prog(n: int, prefix: str, result: dict):
    a = Decimal(random.randint(1, 500)) / 10
    d = Decimal(random.randint(1, 300)) / 10
    for i in range(n):
        result['<%s.%s>' % (prefix, i)] = str(a + i * d)


def difference_prog(n: int, prefix: str, result: dict):
    degree = random.randint(2, 4)
    symbol = sympy.Symbol('x')
    poly = sympy.polys.specialpolys.random_poly(symbol, degree, -5, 5)

    for i in range(n):
        value = poly.subs({symbol: (i+1)}).evalf()
        result['<%s.%s>' % (prefix, i)] = str(poly.subs({symbol: (i+1)}))


def replace_unknown(target: str, prefix: str, result: dict):
    unk_var = random.choice(UNK)
    keys = [key
            for key in result.keys() if key.startswith('<%s.' % target)]
    tgt_var = random.choice(keys)

    result[tgt_var] = unk_var
    result['<%s.0>' % prefix] = unk_var


def rep_prog(size: int, prefix: str, result: dict):
    for i in range(size):
        result['<%s.%s>' % (prefix, i)] = str(Decimal(random.randrange(200)) / 100)


def unk_digit_equation(digit1: int, digit2: int, operator: str, eq_prefix: str, unk_prefix: str, result: dict):
    assert operator in '+-'
    min_digit = min(digit1, digit2)
    a = random.randint(10 ** (digit1 - 1), 10 ** digit1 - 1)
    b = random.randint(10 ** (digit2 - 1), 10 ** digit2 - 1)
    r = a + b

    if operator == '-':
        # a-b = r == r-a = b
        a, b, r = r, a, b

    terms = [list(str(a)), list(str(b)), list(str(r))]
    digits = [-d for d in range(1, min_digit + 1)]
    if min_digit > 2:
        random.shuffle(digits)
        digits = digits[:random.randint(2, min_digit)]

    unknowns = UNK.copy()
    random.shuffle(unknowns)
    unknowns = unknowns[:len(digits)]
    for p, var in zip(digits, unknowns):
        choice = random.randint(0, 2)
        terms[choice][p] = var

    equation = '%s%s%s=%s' % (''.join(terms[0]), operator, ''.join(terms[1]), ''.join(terms[2]))
    result['<%s.0>' % eq_prefix] = equation
    result['<%s.0>' % unk_prefix] = random.choice(unknowns)


def interchange_pair(digit: int, pos1: int, pos2: int, prefix: str, result: dict):
    number = list(str(random.randint(10 ** (digit - 1), 10 ** digit - 1)))

    pos_digit1 = int(math.log10(pos1)) + 1
    pos_digit2 = int(math.log10(pos2)) + 1
    if pos_digit1 > pos_digit2:
        pos_digit1, pos_digit2 = pos_digit2, pos_digit1

    assert 0 < pos_digit1 <= digit
    assert 0 < pos_digit2 <= digit
    assert pos_digit1 != pos_digit2

    if pos_digit2 == digit and number[-pos_digit1] == '0':
        number[-pos_digit1] = str(random.randint(1, 9))

    if number[-pos_digit1] == number[-pos_digit2]:
        digits = [str(i) for i in range(10)]
        if pos_digit2 == digit:
            del digits[0]

        digits = [i for i in digits if i != number[-pos_digit1]]
        number[-pos_digit2] = random.choice(digits)

    interchanged = number.copy()
    interchanged[-pos_digit1], interchanged[-pos_digit2] = interchanged[-pos_digit2], interchanged[-pos_digit1]

    result['<%s.0>' % prefix] = ''.join(number)
    result['<%s.1>' % prefix] = ''.join(interchanged)


def errorpair(digit: int, pos: int, prefix: str, result: dict):
    number = list(str(random.randint(10 ** (digit - 1), 10 ** digit - 1)))

    pos_digit = int(math.log10(pos)) + 1
    assert 0 < pos_digit <= digit

    interchanged = number.copy()
    digits = [str(i) for i in range(10)]
    if pos_digit == digit:
        del digits[0]

    digits = [i for i in digits if i != number[-pos_digit]]
    interchanged[-pos_digit] = random.choice(digits)

    result['<%s.0>' % prefix] = ''.join(number)
    result['<%s.1>' % prefix] = ''.join(interchanged)


def make_system(mode: str, num: int, eqn_prefix: str, unk_prefix: str, result: dict):
    assert mode in '+-'

    b = random.randrange(1, 99)
    a = b * num
    r = a + b if mode == '+' else a - b

    unk = UNK.copy()
    random.shuffle(unk)
    unk_1, unk_2 = unk[:2]

    txt_1 = unk_1 + mode + unk_2 + '=' + str(r)
    txt_2 = [unk_1 + '=' + unk_2] + ([unk_2] * (num - 1))
    txt_2 = '+'.join(txt_2)

    result['<%s.0>' % eqn_prefix] = txt_1
    result['<%s.1>' % eqn_prefix] = txt_2
    result['<%s.0>' % unk_prefix] = unk_1
    result['<%s.1>' % unk_prefix] = unk_1


def eval_expression(expr: str, out_prefix: str, result: dict):
    result['<%s.0>' % out_prefix] = str(eval(expr))


__all__ = [
    'arithmetic_prog',
    'difference_prog',
    'replace_unknown',
    'rep_prog',
    'unk_digit_equation',
    'errorpair',
    'make_system'
]
