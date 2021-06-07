# Note: Do not import other libraries here
# (1) Operator keys
NAME = 'name'
ARITY = 'arity'
CONVERT = 'convert'
COMMUTATIVE = 'commutative'
TOPLV = 'top_level'
PRECEDENCE = 'precedence'

# (2) Word information keys
IS_NUM = 'is_num'
IS_VAR = 'is_var'
IS_PROP = 'is_prop'
VALUE = 'value'
WORD = 'word'

# (3) Token for operand sources
SRC_CONSTANT = 0
SRC_NUMBER = 1
SRC_RESULT = 2
SRC_LIST = [SRC_CONSTANT, SRC_NUMBER, SRC_RESULT]

# (4) I/O keys
QUESTION = 'question'
ANSWER = 'answer'
EQUATION = 'equation'
EXECUTION = 'execution'
