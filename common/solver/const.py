# Note: Do not import other libraries here, except common.sys.key
from common.sys.key import *

# (1) Constants
CON_VALUES = []
CON_TOKENS = [str(x) for x in CON_VALUES]
CON_MAX = len(CON_VALUES)

# (2) Operators
OPR_NEW_EQN = '_NEW_EQN'
OPR_NEW_VAR = '_NEW_VAR'
OPR_DONE = '_DONE'

OPR_VALUES = [
    # 1. NEW_EQN()
    {NAME: OPR_NEW_EQN, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 2. NEW_VAR()
    {NAME: OPR_NEW_VAR, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 3. DONE()
    {NAME: OPR_DONE, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 4. = binary
    {NAME: '=', ARITY: 2, COMMUTATIVE: True, TOPLV: True, CONVERT: (lambda *x: NotImplementedError()),
     PRECEDENCE: 1},
    # 5. + binary
    {NAME: '+', ARITY: 2, COMMUTATIVE: True, TOPLV: False, CONVERT: (lambda *x: x[0] + x[1]),
     PRECEDENCE: 2},
    # 6. - binary
    {NAME: '-', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] - x[1]),
     PRECEDENCE: 2},
    # 7. * binary
    {NAME: '*', ARITY: 2, COMMUTATIVE: True, TOPLV: False, CONVERT: (lambda *x: x[0] * x[1]),
     PRECEDENCE: 3},
    # 8. / binary
    {NAME: '/', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] / x[1]),
     PRECEDENCE: 3},
    # 9. ^ binary
    {NAME: '^', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] ** x[1]),
     PRECEDENCE: 4},
    # 10. PRINT(result)
    {NAME: OPR_PRINT, ARITY: 1, COMMUTATIVE: True, TOPLV: True, CONVERT: (lambda *x: print(x)), PRECEDENCE: None},
    # 11. SUM(LIST)
    {NAME: OPR_SUM, ARITY: 1, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: sum(range(x[0],x[1],x[2]))), PRECEDENCE: None},
    # 12. LIST(): create list
    {NAME: OPR_LIST, ARITY: 0, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: list(*x)), PRECEDENCE: None},
    # 13. APPEND(LIST,x)
    {NAME: OPR_APPEND, ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 14. COMB(x,y) 
    {NAME: COMB, ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 15. PERM(x,y)
    {NAME: PERM, ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 16. MIN(x)
    {NAME: MIN, ARITY: 1, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 17. MAX(x)
    {NAME: MAX, ARITY: 1, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 18. RANGE(start, end, step)
    {NAME: RANGE, ARITY: 3, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 19. LCM(LIST)
    {NAME: LCM, ARITY: 1, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 20. GCD(LIST)
    {NAME: GCD, ARITY: 1, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # 21. NO_13(LIST,LIST): function for the 13th question
    {NAME: NO_13, ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    {NAME: , ARITY: , COMMUTATIVE: False, TOPLV: False, CONVERT: None, PRECEDENCE: None},

]
OPR_TOKENS = [x[NAME] for x in OPR_VALUES]
OPR_SZ = len(OPR_TOKENS)

# Special operator ids
OPR_NEW_EQN_ID = OPR_TOKENS.index(OPR_NEW_EQN)
OPR_NEW_VAR_ID = OPR_TOKENS.index(OPR_NEW_VAR)
OPR_DONE_ID = OPR_TOKENS.index(OPR_DONE)
OPR_EQ_SGN_ID = OPR_TOKENS.index('=')
OPR_PLUS_ID = OPR_TOKENS.index('+')

OPR_SPECIAL = OPR_TOKENS[:OPR_EQ_SGN_ID]
OPR_NON_SPECIAL = OPR_TOKENS[OPR_EQ_SGN_ID:]

OPR_MAX_ARITY = max(op[ARITY]
                    for op in OPR_VALUES)
OPR_TOP_LV = {i
              for i, token in enumerate(OPR_VALUES)
              if token[TOPLV]}
