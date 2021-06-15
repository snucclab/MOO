##@@@@ CODE-REPLACEMENT: {result} by {result}_pycode ##
_MULTIDIGIT = re.compile('[0-9][A-Z]|[A-Z][0-9]|[A-Z]{2,}')
if any(_MULTIDIGIT.match(eq) is not None for eq in {LIST}):
    # Multi-digit case
    for eq in {LIST}:
        cur_list = re.split("[-+/*=]", eq)

else:
    _TRANSFORMATIONS = sympy.parsing.sympy_parser.standard_transformations + (sympy.parsing.sympy_parser.convert_equals_signs,)
    equations = []

    for eq in {LIST}:
        lhs, rhs = eq.split('=')
        eq = sympy.parse_expr(eq, transformations=_TRANSFORMATIONS)
        equations.append(eq)


sympy.parse_expr(
vars = set()
p = re.compile("[=+-\/*]")
for i in {LIST}:
    cur_list = p.split(i)
    for el in cur_list:
        if el.isdigit() == False:
            vars.add(el)

for i in list(vars):
    globals()[i] = sympy.symbols(i, real=True)

system = []
for expr in {LIST}:
    split_expr = expr.split('=')
    left = split_expr[0]
    right = split_expr[1]
    system.append( 'sympy.Eq(' + left + ',' + right + ')' )

sol_dict = sympy.solve([eval(system[0]),eval(system[1])], dict=True)
{result} = sol_dict[0][eval(str({target}))]
print({result}, end="")
## CODE-REPLACEMENT END for {result} @@@@##
