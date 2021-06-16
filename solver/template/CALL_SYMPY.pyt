##@@@@ CODE-REPLACEMENT: {result} by {result}_pycode ##
_MULTIDIGIT = re.compile('.*([0-9][A-Z]|[A-Z][0-9]|[A-Z]{{2,}}).*')
_DIGITNUM = re.compile('^\\d+$')
_TRANSFORMATIONS = sympy.parsing.sympy_parser.standard_transformations + (sympy.parsing.sympy_parser.convert_equals_signs,)

is_multidigit = False
for eq in {LIST}:
    if _MULTIDIGIT.match(eq) is not None:
        is_multidigit = True
        break

_result = dict()
if is_multidigit:
    # Multi-digit case
    assert len({LIST}) == 1
    eq = {LIST}[0]
    a, op1, b, op2, r = re.split('([-+=])', eq)
    if op1 == '=':
        # Make as a +/- b = c
        a, op1, b, op2, r = b, op2, r, op1, a

    _pycodes = []
    max_len = max(len(a), len(b), len(r))
    carry = 0
    for i in range(1, max_len + 1):
        a_i = a[-i] if len(a) >= i else '0'
        b_i = b[-i] if len(b) >= i else '0'
        r_i = r[-i] if len(r) >= i else '0'

        abr_i = a_i + b_i + r_i
        equation = '%s%s%s+(%s)=%s' % (a_i, op1, b_i, carry, r_i)
        equation = equation.replace('E', '_E')

        if _DIGITNUM.fullmatch(abr_i) is not None:
            # Digit case
            if eval(equation.replace('=', '==')):
                # No carry occurred
                carry = 0
            elif eval(equation.replace('=', '<')):
                # 계산결과가 실제보다 작음. 자리내림 필요
                carry = -1
            else:
                # 계산결과가 실제보다 큼. 자리올림 필요
                carry = +1
            continue

        sympy_eq = sympy.parse_expr(equation, transformations=_TRANSFORMATIONS)
        solutions = sympy.solve(sympy_eq, dict=True)
        digit_key, digit_value = [(key.name if key.name != '_E' else 'E', value) for key, value in solutions[0].items()][0]

        if digit_value < 0:
            # 자리내림/올림 필요
            carry = -1 if op1 == '-' else +1
            digit_value += 10
        elif digit_value >= 10:
            # 자리내림/올림 필요
            carry = -1 if op1 == '-' else +1
            digit_value -= 10
        else:
            carry = 0

        _result[digit_key] = int(digit_value)
        _pycodes.append('# solution of %s\n{result}_%s=%s' % (equation, digit_key, digit_value))

    {result}_pycode = '\n'.join(_pycodes) + '\n{result} = {result}_%s\n' % {target}
    {result} = _result[{target}]
else:
    equations = []

    for eq in {LIST}:
        # E 처리
        eq = eq.replace('E', '_E')
        eq = sympy.parse_expr(eq, transformations=_TRANSFORMATIONS)
        equations.append(eq)

    _result = sympy.solve(equations, dict=True)
    _result = {{(key.name if key.name != '_E' else 'E'): value for key, value in _result[0].items()}}

    {result} = _result[{target}]
    is_int = {result}.is_integer
    if is_int:
        {result} = int({result})
    else:
        {result} = float({result})

    {result}_pycode = '# solution of %s\n' % (', '.join({LIST})) + \
        '\n'.join(['{result}_%s = %s' % t for t in _result.items()]) + \
        '\n{result} = %s({result}_%s)\n' % ('int' if is_int else 'float', {target})
## CODE-REPLACEMENT END for {result} @@@@##
