
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
    {NAME: OPR_NEW_EQN, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 2. NEW_VAR()
    {NAME: OPR_NEW_VAR, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 3. DONE()
    {NAME: OPR_DONE, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 4. EQ(float, float)
    {NAME: EQ, ARITY: 2, COMMUTATIVE: True, ISVOID: True, CONVERT: (lambda *x: NotImplementedError()),
     PRECEDENCE: 1},
    # 5. +(float, float)
    {NAME: '+', ARITY: 2, COMMUTATIVE: True, ISVOID: False, CONVERT: (lambda *x: x[0] + x[1]),
     PRECEDENCE: 2},
    # 6. -(float, float)
    {NAME: '-', ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] - x[1]),
     PRECEDENCE: 2},
    # 7. *(float, float)
    {NAME: '*', ARITY: 2, COMMUTATIVE: True, ISVOID: False, CONVERT: (lambda *x: x[0] * x[1]),
     PRECEDENCE: 3},
    # 8. /(float,float)
    {NAME: '/', ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] / x[1]),
     PRECEDENCE: 3},
    # 9. ^(float,float)
    {NAME: '^', ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda *x: x[0] ** x[1]),
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
    {NAME: COMB, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 15. PERM(int,int)
    {NAME: PERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 16. MIN(List)
    {NAME: MIN, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 17. MAX(List)
    {NAME: MAX, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 18. RANGE(start: int, end: int, step: int)
    {NAME: RANGE, ARITY: 3, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 19. LCM(List)
    {NAME: LCM, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 20. GCD(List)
    {NAME: GCD, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 21. COUNT_MULTI(List,List): function for the 13th question
    {NAME: NO_13, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 22. DIGIT(int, digit: int)
    {NAME: DIGIT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 23. TO_INT(float)
    {NAME: TO_INT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 24. CALL_SYMPY(List)
    {NAME: CALL_SYMPY, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 25. REVERSE_DIGIT(int)
    {NAME: REVERSE_DIGIT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 26. SEQ_TERM(List, int)
    {NAME: SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 27. REP_SEQ_TERM(List, int)
    {NAME: REP_SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 28. MAKE_PAIR(str, str_or_int)
    {NAME: MAKE_PAIR, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 29. COUNT(List) 
    {NAME: COUNT, ARITY: 1, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 30. LT(List, int)
    {NAME: LT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 31. LE(List, int)
    {NAME: LE, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 32. GT(List, int)
    {NAME: GT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 33. GE(List, int)
    {NAME: GE, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None}
    # 34. LIST_CONCAT(ls1: List[Any], ls2: List[Any]) -> List[Any]
    {NAME: LIST_CONCAT, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None}
    # 35. LIST_INDEX(ls: List[Union[int,float,str]],item: Union[str, int, float]) -> int
    {NAME: LIST_INDEX, ARITY: 2, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None}
    # 36. LIST_REPLACE(ls: List[Union[int,float,str]], n: int, item: Union[int,float,str]) -> List[Union[int,float,str]]
    {NAME: LIST_REPLACE, ARITY: 3, COMMUTATIVE: False, ISVOID: False, CONVERT: None, PRECEDENCE: None}
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
