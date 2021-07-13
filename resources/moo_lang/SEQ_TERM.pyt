##@@@@ CODE-REPLACEMENT: {result} by {result}_pycode ##
data_points = []
for idx, value in enumerate({seq}):
    if isinstance(value, (int, float)):
        data_points.append((idx+1, value))

symbol = sympy.Symbol('n', real=True)
general_term = sympy.polys.polyfuncs.interpolate(data_points, symbol)
# Clean-up decimals
cleaned_term = 0
for i, c in enumerate(reversed(sympy.Poly(general_term, symbol).all_coeffs())):
    if isinstance(c, (sympy.Rational, sympy.Integer)):
        cleaned_term += c * (symbol ** i)
    else:
        cleaned_term += float('%.5f' % c) * (symbol ** i)
{result}_pycode = '\n{result} = %s\n' % str(cleaned_term).replace('n', str({index}))  # Make python code
_result = cleaned_term.subs({{symbol: {index} }})
if _result.is_integer:
    {result} = int(_result)
else:
    {result} = float(_result)
## CODE-REPLACEMENT END for {result} @@@@##
