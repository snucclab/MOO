import sympy
import re
from typing import List, Any

def call_sympy(prob: List, target: Any) -> Union[int, float]:
    vars = set()
    p = re.compile("[=+-\/*]")
    for i in prob:
        cur_list = p.split(i)
        for el in cur_list:
            if el.isdigit() == False:
                vars.add(el)

    for i in list(vars):
        globals()[i] = sympy.symbols(i, real=True)

    system = []
    for expr in prob:
        split_expr = expr.split('=')
        left = split_expr[0]
        right = split_expr[1]
        system.append( 'sympy.Eq(' + left + ',' + right + ')' )

    sol_dict = sympy.solve([eval(system[0]),eval(system[1])], dict=True)
    result = sol_dict[0][eval(str(target))]

    return result
