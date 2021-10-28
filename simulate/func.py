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

def unk_digit_equation_multi(num_unk: int, num_digit: int, operator: str, result: dict):
    # num_unk는 미지수 개수 num_digit은 몇자리수인지
    # ex. num_unk=2, num_digit=3 -> 세자리 수 두 개를 더하는데 A34 + 3B5 = 999이다. 뭐랑 뭐를 구하여라.
    assert operator in '+-'
    UNK = [chr(ord('A') + i) for i in range(26)]
    unk_list = []
    # 이하 포문 미지수 샘플링 역할
    # UNK가 사용될 num_unk개의 미지수
    for i in range(num_unk):
        n = random.randint(0, len(UNK)-1)
        unk_list.append(UNK[n])
        UNK.pop(n)

    while True:
        a = random.randint(10**(num_digit-1), 10**num_digit)
        b = random.randint(10**(num_digit-1), 10**num_digit)
        r = a + b
        if 10**(num_digit-1) <= r < 10**num_digit:
            break
    if operator == '-':
        a, b, r = r, a, b
    terms = [list(str(a)), list(str(b)), list(str(r))]
    num_list = []
    for t in terms:
        for a in t:
            num_list.append(a)
    num_list = list(set(num_list))

    equation = '%s%s%s=%s' % (''.join(terms[0]), operator, ''.join(terms[1]), ''.join(terms[2]))
    countunk = 0
    for n in unk_list.copy():
        sample = random.randint(0, len(num_list)-1)
        equation = equation.replace(num_list[sample], n)
        num_list.pop(num_list.index(num_list[sample]))
        unk_list.pop(unk_list.index(n))
    for token in equation:
        if token in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            result['<%s.%s>' % ('unknown', countunk)] = token
            countunk+=1

    result['<%s.0>' % 'equation'] = equation

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
    result['<%s.1>' % unk_prefix] = unk_2


def eval_expression(expr: str, out_prefix: str, result: dict):
    result['<%s.0>' % out_prefix] = str(eval(expr))


def different_number(begin: int, to: int, size: int, prefix: str, result: dict):
    items = list(range(begin, to))
    random.shuffle(items)
    items = items[:size]

    for i, n in enumerate(items):
        result['<%s.%s>' % (prefix, i)] = str(n)


def div_to_int(start: int, end: int, result: dict):
    # divisor는 start와 end 사이의 랜덤 숫자
    divisor = Decimal(random.randint(start, end))
    # divisor의 약수들
    diviser = []
    for i in range(1, int(divisor+1)):
        if divisor % i == 0:
            diviser.append(Decimal(i))

    result['<%s.%s>' % ("divisor", 0)] = str(divisor)
    # diviser는 약수 목록 중 랜덤하게 하나 뽑은 것
    result['<%s.%s>' % ("diviser", 0)] = str(random.choice(diviser))


# must not start with prime number..!
def make_fraction(start: int, end: int, fc: int, result: dict):
    # random choice num not be prime number
    # fc개 만큼 분수 만듦
    num_list = [0] + [i for i in range(1, end + 1)]  # 0에서 end까지 [0, 1, ..., end]
    for i, num in enumerate(num_list):  # 합성수 제거
        if num == 0 or num == 1:
            continue
        index = 2
        while i * index < len(num_list):
            num_list[i * index] = 0
            index += 1

    prime_list = [i for i in num_list if i != 0]  # 소수 list
    candidate_num_list = list(range(0, end))  # 다시 0부터 end까지 list

    for prime in sorted(prime_list, reverse=True):  # 소수 내림차순으로
        del candidate_num_list[prime]  # 후보에서 소수 없애고
    candidates = candidate_num_list[candidate_num_list.index(start):]  # start부터로 끊어줌

    # find fraction
    num = random.choice(candidates)   # 후보 중에 하나 랜덤으로 뽑기
    temp = num
    divisers = []  # 후보의 약수들에서 1이랑 자기자신 제외

    diviser = Decimal(2)  # 2부터
    while temp != Decimal(1):
        if temp % diviser == Decimal(0):
            divisers.append(Decimal(diviser))
            temp /= diviser
        else:
            diviser += Decimal(1)

    if len(divisers) >= 2:
        divisers = random.sample(divisers, 2)
    else:
        divisers.append(divisers[0])

    result['<%s.%s>' % ("num", 0)] = str(num)
    for i in range(fc):
        result['<%s.%s>' % ("fraction", i)] = str(random.randint(1,int(divisers[i])-1)) + '/' + str(divisers[i])


def round_up(digit: int, pos: int, flag: str, result: dict):
    # round_digit max min count에 따라서 각각 int로 저장
    before_round = random.randint(1 * (10 ** (digit - 1)), 1 * (10 ** digit) - 1)
    after_round = before_round
    round_digit = int(str(before_round)[digit - len(str(pos))])

    if round_digit < 5: # 반올림(버림)
        round_digit = [0, 1, 2, 3, 4]
        after_round -= before_round % (pos * 10)
        if flag=='max':
            result['<%s.%s>' % ("roundDigit", 0)] = str(4)
        elif flag=='min':
            result['<%s.%s>' % ("roundDigit", 0)] = str(0)
    else: # 반올림(올림)
        round_digit = [5, 6, 7, 8, 9]
        after_round -= before_round % (pos * 10)
        after_round += (pos * 10)
        if flag=='max':
            result['<%s.%s>' % ("roundDigit", 0)] = str(9)
        elif flag=='min':
            result['<%s.%s>' % ("roundDigit", 0)] = str(5)

    result['<%s.%s>' % ("beforeRound", 0)] = str(before_round)
    result['<%s.%s>' % ("afterRound", 0)] = str(after_round)
    if flag=='count':
        result['<%s.%s>' % ("roundDigit", 0)] = str(5)

def make_triangle(result: dict):
    while True:
        a = random.randint(1,50)
        b = random.randint(1,50)
        c = random.randint(1,50)
        if (a<b) and (b<c) and (c<a+b):
            break
    result['<%s.%s>' % ("triangle", 0)] = str(a)
    result['<%s.%s>' % ("triangle", 1)] = str(b)
    result['<%s.%s>' % ("triangle", 2)] = str(c)


__all__ = [
    'arithmetic_prog',
    'difference_prog',
    'replace_unknown',
    'rep_prog',
    'unk_digit_equation_multi',
    'unk_digit_equation',
    'errorpair',
    'make_system',
    'div_to_int',
    'make_fraction',
    'round_up',
    'make_triangle'
]
