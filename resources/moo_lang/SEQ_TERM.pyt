##@@@@ CODE-REPLACEMENT: {result} by {result}_pycode ##
data_points = []
for idx, value in enumerate({seq}):
    if isinstance(value, (int, float)):
        data_points.append((idx+1, value))

symbol = sympy.Symbol('n', real=True)
general_term = sympy.polys.polyfuncs.interpolate(data_points, symbol)
# Clean-up decimals
general_coeffs = [float('%.5f' % c) for c in sympy.Poly(general_term, symbol).all_coeffs()]
general_term = sum([c * (symbol ** n) for n, c in enumerate(reversed(general_coeffs))])
{result}_pycode = '\n{result} = %s\n' % str(general_term).replace('n', str({index}))  # Make python code
_result = general_term.subs({{symbol: {index} }})
if _result.is_integer:
    {result} = int(_result)
else:
    {result} = float(_result)
## CODE-REPLACEMENT END for {result} @@@@##
