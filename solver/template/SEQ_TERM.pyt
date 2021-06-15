##@@@@ CODE-REPLACEMENT: {result} by {result}_pycode ##
data_points = []
for idx, value in enumerate({seq}):
    if isinstance(value, (int, float)):
        data_points.append((idx+1, value))

symbol = sympy.Symbol('n', real=True)
general_term = sympy.polys.polyfuncs.interpolate(data_points, symbol)
{result}_pycode = str(general_term).replace('n', str({index}))
_result = general_term.subs({{symbol: {index} }})
if _result.is_integer:
    {result} = int(_result)
else:
    {result} = float(_result)
## CODE-REPLACEMENT END for {result} @@@@##
