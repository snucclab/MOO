# -------------------------------------------------------
# -------------------- OPERANDS -------------------------
# -------------------------------------------------------

# (1) Constants
CON_VALUES = []
CON_MAX = len(CON_VALUES)

# (2)

# Maximum capacity of variable, numbers and expression memories
# Format of variable/number/expression tokens
PREFIX_LEN = 2
VAR_PREFIX = 'X_'
NUM_PREFIX = 'N_'
RES_PREFIX = 'R_'
VAR_FORMAT = '%s%%02d' % VAR_PREFIX
NUM_FORMAT = '%s%%02d' % NUM_PREFIX
RES_FORMAT = '%s%%02d' % RES_PREFIX
assert len(VAR_PREFIX) == PREFIX_LEN and len(NUM_PREFIX) == PREFIX_LEN and len(RES_PREFIX) == PREFIX_LEN


# Number of all operand tokens
CON_END = len(CON_TOKENS)
NUM_BEGIN = CON_END
NUM_END = NUM_BEGIN + NUM_MAX
RES_BEGIN = NUM_END

# Token for operand sources
SRC_CONSTANT = 0
SRC_NUMBER = 1
SRC_RESULT = 2
SRC_LIST = [SRC_CONSTANT, SRC_NUMBER, SRC_RESULT]

from sympy import Eq

# Operator tokens
OPR_TOKENS = ['__NEW_EQN', '__NEW_VAR', '__DONE', '=', '+', '-', '*', '/', '^']
OPR_SZ = len(OPR_TOKENS)

# Special operator ids
OPR_NEW_EQN_ID = 0
OPR_NEW_VAR_ID = OPR_NEW_EQN_ID + 1
OPR_DONE_ID = OPR_NEW_VAR_ID + 1
OPR_EQ_SGN_ID = OPR_DONE_ID + 1
OPR_PLUS_ID = OPR_TOKENS.index('+')
OPR_NEW_EQN = OPR_TOKENS[OPR_NEW_EQN_ID]
OPR_NEW_VAR = OPR_TOKENS[OPR_NEW_VAR_ID]
OPR_DONE = OPR_TOKENS[OPR_DONE_ID]

OPR_SPECIAL = OPR_TOKENS[:OPR_EQ_SGN_ID]
OPR_NON_SPECIAL = OPR_TOKENS[OPR_EQ_SGN_ID:]

# Operator information
KEY_ARITY = 'arity'
KEY_CONVERT = 'convert'
KEY_COMMUTATIVE = 'commutative'
KEY_TOPLV = 'top_level'
KEY_PRECEDENCE = 'precedence'
OPR_VALUES = [
    # NEW_EQN()
    {KEY_ARITY: 0, KEY_COMMUTATIVE: True, KEY_TOPLV: False, KEY_CONVERT: None, KEY_PRECEDENCE: None},
    # NEW_VAR()
    {KEY_ARITY: 0, KEY_COMMUTATIVE: True, KEY_TOPLV: False, KEY_CONVERT: None, KEY_PRECEDENCE: None},
    # DONE()
    {KEY_ARITY: 0, KEY_COMMUTATIVE: True, KEY_TOPLV: False, KEY_CONVERT: None, KEY_PRECEDENCE: None},
    # = binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: True, KEY_TOPLV: True, KEY_CONVERT: (lambda *x: Eq(x[0], x[1], evaluate=False)),
     KEY_PRECEDENCE: 1},
    # + binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: True, KEY_TOPLV: False, KEY_CONVERT: (lambda *x: x[0] + x[1]),
     KEY_PRECEDENCE: 2},
    # - binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: False, KEY_TOPLV: False, KEY_CONVERT: (lambda *x: x[0] - x[1]),
     KEY_PRECEDENCE: 2},
    # * binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: True, KEY_TOPLV: False, KEY_CONVERT: (lambda *x: x[0] * x[1]),
     KEY_PRECEDENCE: 3},
    # / binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: False, KEY_TOPLV: False, KEY_CONVERT: (lambda *x: x[0] / x[1]),
     KEY_PRECEDENCE: 3},
    # ^ binary
    {KEY_ARITY: 2, KEY_COMMUTATIVE: False, KEY_TOPLV: False, KEY_CONVERT: (lambda *x: x[0] ** x[1]),
     KEY_PRECEDENCE: 4}
]
assert OPR_SZ == len(OPR_VALUES)

OPR_MAX_ARITY = max(op[KEY_ARITY] for op in OPR_VALUES)
OPR_TOP_LV = {i
              for i, token in enumerate(OPR_VALUES) if token[KEY_TOPLV]}
