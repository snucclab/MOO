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
    # NEW_EQN()
    {NAME: OPR_NEW_EQN, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # NEW_VAR()
    {NAME: OPR_NEW_VAR, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # DONE()
    {NAME: OPR_DONE, ARITY: 0, COMMUTATIVE: True, TOPLV: False, CONVERT: None, PRECEDENCE: None},
    # = binary
    {NAME: '=', ARITY: 2, COMMUTATIVE: True, TOPLV: True, CONVERT: (lambda *x: NotImplementedError()),
     PRECEDENCE: 1},
    # + binary
    {NAME: '+', ARITY: 2, COMMUTATIVE: True, TOPLV: False, CONVERT: (lambda *x: x[0] + x[1]),
     PRECEDENCE: 2},
    # - binary
    {NAME: '-', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] - x[1]),
     PRECEDENCE: 2},
    # * binary
    {NAME: '*', ARITY: 2, COMMUTATIVE: True, TOPLV: False, CONVERT: (lambda *x: x[0] * x[1]),
     PRECEDENCE: 3},
    # / binary
    {NAME: '/', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] / x[1]),
     PRECEDENCE: 3},
    # ^ binary
    {NAME: '^', ARITY: 2, COMMUTATIVE: False, TOPLV: False, CONVERT: (lambda *x: x[0] ** x[1]),
     PRECEDENCE: 4}
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
