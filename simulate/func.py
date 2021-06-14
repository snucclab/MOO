import random
from decimal import Decimal
unk = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def arithmetic_prog(n: int):
    a = random.randint(1, 500) / 10
    d = random.randint(1, 300) / 10
    prog = []
    for i in range(n):
        prog.append(a+i*d)
    return prog

print(arithmetic_prog(7))

def difference_prog(n: int):
    a = random.randint(1, 500) / 10
    b = arithmetic_prog(n)
    prog = []
    a_n = a
    for i in range(n):
        a_n += b[i]
        prog.append(a_n)
    return prog

print(difference_prog(6))

def replace_unknown(prog: list):
    idx = random.randrange(len(prog))
    unk_idx = random.randrange(len(unk))
    prog[idx] = unk[unk_idx]
    return prog

print(replace_unknown(difference_prog(7)))

def rep_prog():
    prog = []
    for i in range(random.randint(3, 7)):
        prog.append(random.randrange(200) / 10)
    prog += prog
    return prog

print(rep_prog())

def diff_two(num0: int):
    if (num0-2) < 0:
        return num0+2
    else:
        return num0+random.choice((2, -2))

print(diff_two(6))

def gen_expr(operation):
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    unk = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    list = [a, b]
    new_list = []
    idx = [0, 1]
    if operation == 'sum':
        s = sum(list)
        for i in list:
            txt = str(i)
            txt = txt.replace(txt[idx.pop(random.randrange(len(idx)))], unk.pop(random.randrange(len(unk))))
            new_list.append(txt)
        result = new_list[0]+'+'+new_list[1]+'='+str(s)
        return result
    elif operation == 'diff':
        list.sort()
        list.reverse()
        d = list[0]-list[1]
        for i in list:
            txt = str(i)
            txt = txt.replace(txt[idx.pop(random.randrange(len(idx)))], unk.pop(random.randrange(len(unk))))
            new_list.append(txt)
        result = new_list[0]+'-'+new_list[1]+'='+str(d)
        return result

print(gen_expr('sum'))
print(gen_expr('diff'))

def errorpair(digits: int):
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    orig = str(nums.pop(nums.index(random.choice(nums))))
    if digits > 2:
        nums.append(0)
    for i in range(digits-1):
        num = nums.pop(nums.index(random.choice(nums)))
        orig += str(num)
    miss = []
    for i in orig:
        miss.append(i)
    random.shuffle(miss)
    err = ''.join(miss)
    while (err == orig) or (err[0] == '0'):
        random.shuffle(miss)
        err = ''.join(miss)
    return int(orig), int(err)

print(errorpair(3))

def make_unk_pair(mode: str, num: int):
    unk = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z']
    b = random.randrange(1, 99)
    a = b*num
    plus = a+b
    minus = a-b
    if mode == 'plus':
        unk_1 = unk.pop(unk.index(random.choice(unk)))
        unk_2 = unk.pop(unk.index(random.choice(unk)))
        txt_1 = unk_1+'+'+unk_2+'='+str(plus)
        txt_2 = unk_1+'='+unk_2
        for i in range(num-1):
            txt_2 += '+'+unk_2
        return txt_1, txt_2
    if mode == 'minus':
        unk_1 = unk.pop(unk.index(random.choice(unk)))
        unk_2 = unk.pop(unk.index(random.choice(unk)))
        txt_1 = unk_1 + '-' + unk_2 + '=' + str(minus)
        txt_2 = unk_1 + '=' + unk_2
        for i in range(num - 1):
            txt_2 += '+' + unk_2
        return txt_1, txt_2

print(make_unk_pair('minus', 5))
