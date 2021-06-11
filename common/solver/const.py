
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
OPR_EQ = 'EQ'
OPR_ADD = 'ADD'
OPR_SUB = 'SUB'
OPR_MUL = 'MUL'
OPR_DIV = 'DIV'
OPR_POW = 'POW'
OPR_PRINT = 'PRINT'
OPR_SUM = 'SUM'
OPR_LIST = 'LIST'
OPR_APPEND = 'APPEND'
OPR_COMB = 'COMB'
OPR_PERM = 'PERM'
OPR_MIN = 'MIN'
OPR_MAX = 'MAX'
OPR_RANGE = 'RANGE'
OPR_LCM = 'LCM'
OPR_GCD = 'GCD'
OPR_COUNT_MULTI = 'COUNT_MULTI'
OPR_DIGIT = 'DIGIT'
OPR_TO_INT = 'TO_INT'
OPR_CALL_SYMPY = 'CALL_SYMPY'
OPR_REVERSE_DIGIT = 'REVERSE_DIGIT'
OPR_SEQ_TERM = 'SEQ_TERM'
OPR_REP_SEQ_TERM = 'REP_SEQ_TERM'
OPR_MAKE_PAIR = 'MAKE_PAIR'
OPR_COUNT = 'COUNT'
OPR_LT = 'LT'
OPR_LE = 'LE'
OPR_GT = 'GT'
OPR_GE = 'GE'
OPR_LIST_CONCAT = 'LIST_CONCAT'
OPR_LIST_INDEX = 'LIST_INDEX'
OPR_LIST_REPLACE = 'LIST_REPLACE'

OPR_VALUES = [
    # 1. NEW_EQN()
    {NAME: OPR_NEW_EQN, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 2. NEW_VAR()
    {NAME: OPR_NEW_VAR, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 3. DONE()
    {NAME: OPR_DONE, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 4. EQ(float, float)
    {NAME: OPR_EQ, ARITY: 2, COMMUTATIVE: True, ISVOID: True, CONVERT: (lambda *x: NotImplementedError()),
     PRECEDENCE: 1},
    # 5. ADD(float, float)
    {NAME: OPR_ADD, ARITY: 2, COMMUTATIVE: True, ISVOID: False, CONVERT: (lambda *x: x[0] + x[1]),
     PRECEDENCE: 2},
    # 6. SUB(float, float)
    {NAME: OPR_SUB, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] - x[1]),
     PRECEDENCE: 2},
    # 7. MUL(float, float)
    {NAME: OPR_MUL, ARITY: 2, COMMUTATIVE: True, ISVOID: False, CONVERT: (lambda *x: x[0] * x[1]),
     PRECEDENCE: 3},
    # 8. DIV(float,float)
    {NAME: OPR_DIV, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] / x[1]),
     PRECEDENCE: 3},
    # 9. POW(float,float)
    {NAME: OPR_POW, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] ** x[1]),
     PRECEDENCE: 4},
    # 10. PRINT(string_or_int_or_float)
    {NAME: OPR_PRINT, ARITY: 1, COMMUTATIVE: True, ISVOID: True, CONVERT: (lambda *x: print(x)), PRECEDENCE: None},
    # 11. SUM(List)
    {NAME: OPR_SUM, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: sum(range(x[0],x[1],x[2]))), PRECEDENCE: None},
    # 12. LIST(): create list
    {NAME: OPR_LIST, ARITY: 0, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: list(*x)), PRECEDENCE: None},
    # 13. APPEND(List,Any)
    {NAME: OPR_APPEND, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 14. COMB(int,int) 
    {NAME: OPR_COMB, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 15. PERM(int,int)
    {NAME: OPR_PERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 16. MIN(List)
    {NAME: OPR_MIN, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 17. MAX(List)
    {NAME: OPR_MAX, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 18. RANGE(start: int, end: int, step: int)
    {NAME: OPR_RANGE, ARITY: 3, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 19. LCM(List)
    {NAME: OPR_LCM, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 20. GCD(List)
    {NAME: OPR_GCD, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 21. COUNT_MULTI(List,List): function for the 13th question
    {NAME: OPR_COUNT_MULTI, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 22. DIGIT(int, digit: int)
    {NAME: OPR_DIGIT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 23. TO_INT(float)
    {NAME: OPR_TO_INT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 24. CALL_SYMPY(List)
    {NAME: OPR_CALL_SYMPY, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 25. REVERSE_DIGIT(int)
    {NAME: OPR_REVERSE_DIGIT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 26. SEQ_TERM(List, int)
    {NAME: OPR_SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 27. REP_SEQ_TERM(List, int)
    {NAME: OPR_REP_SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 28. MAKE_PAIR(str, str_or_int)
    {NAME: OPR_MAKE_PAIR, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 29. COUNT(List) 
    {NAME: OPR_COUNT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 30. LT(List, int)
    {NAME: OPR_LT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 31. LE(List, int)
    {NAME: OPR_LE, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 32. GT(List, int)
    {NAME: OPR_GT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 33. GE(List, int)
    {NAME: OPR_GE, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 34. LIST_CONCAT(ls1: List[Any], ls2: List[Any]) -> List[Any]
    {NAME: OPR_LIST_CONCAT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 35. LIST_INDEX(ls: List[Union[int,float,str]],item: Union[str, int, float]) -> int
    {NAME: OPR_LIST_INDEX, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 36. LIST_REPLACE(ls: List[Union[int,float,str]], n: int, item: Union[int,float,str]) -> List[Union[int,float,str]]
    {NAME: OPR_LIST_REPLACE, ARITY: 3, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None}
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
              if token[ISVOID]}
