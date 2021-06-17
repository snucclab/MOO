# Note: Do not import other libraries here, except common.sys.key
from common.sys.key import *

# (1) Constants
CON_VALUES = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '12'
]
CON_TOKENS = [str(x) for x in CON_VALUES]
CON_MAX = len(CON_VALUES)

# (2) Operators
OPR_NEW_EQN = '_NEW_EQN'
OPR_DONE = '_DONE'
OPR_EQ = 'EQ'
OPR_ADD = 'ADD'
OPR_SUB = 'SUB'
OPR_MUL = 'MUL'
OPR_DIV = 'DIV'
OPR_MOD = 'MOD'
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
OPR_CEIL = 'CEIL'
OPR_LIST_MUL = 'LIST_MUL'
OPR_CHANGE_DIGIT = 'CHANGE_DIGIT'
OPR_GET_DIGIT = 'GET_DIGIT'

OPR_VALUES = [
    # 1. NEW_EQN()
    {NAME: OPR_NEW_EQN, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 3. DONE()
    {NAME: OPR_DONE, ARITY: 0, COMMUTATIVE: True, ISVOID: False, CONVERT: None, PRECEDENCE: None},
    # 4. EQ(float, float)
    {NAME: OPR_EQ, ARITY: 2, COMMUTATIVE: True, ISVOID: True,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 5. ADD(float, float)
    {NAME: OPR_ADD, ARITY: 2, COMMUTATIVE: True, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 6. SUB(float, float)
    {NAME: OPR_SUB, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 7. MUL(float, float)
    {NAME: OPR_MUL, ARITY: 2, COMMUTATIVE: True, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 8. DIV(float,float)
    {NAME: OPR_DIV, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 9. MOD(int, int)
    {NAME: OPR_MOD, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 10. POW(float,float)
    {NAME: OPR_POW, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'x2': x[1], 'result': res}), PRECEDENCE: None},
    # 11. PRINT(string_or_int_or_float)
    {NAME: OPR_PRINT, ARITY: 1, COMMUTATIVE: True, ISVOID: True,
     CONVERT: (lambda res, *x: {'value': x[0], 'result': res}), PRECEDENCE: None},
    # 12. SUM(List)
    {NAME: OPR_SUM, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'lst': x[0], 'result': res}), PRECEDENCE: None},
    # 13. LIST(): create list
    {NAME: OPR_LIST, ARITY: 0, COMMUTATIVE: False, ISVOID: False, CONVERT: (lambda res, *x: {'result': res}),
     PRECEDENCE: None},
    # 14. APPEND(List,Any)
    {NAME: OPR_APPEND, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'lst': x[0], 'x1': x[1], 'result': res}), PRECEDENCE: None},
    # 15. COMB(int,int) 
    {NAME: OPR_COMB, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'n': x[0], 'k': x[1], 'result': res}), PRECEDENCE: None},
    # 16. PERM(int,int)
    {NAME: OPR_PERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'n': x[0], 'k': x[1], 'result': res}), PRECEDENCE: None},
    # 17. MIN(List)
    {NAME: OPR_MIN, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'LIST': x[0], 'result': res}), PRECEDENCE: None},
    # 18. MAX(List)
    {NAME: OPR_MAX, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'LIST': x[0], 'result': res}), PRECEDENCE: None},
    # 19. RANGE(start: int, end: int, step: int)
    {NAME: OPR_RANGE, ARITY: 3, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'start': x[0], 'end': x[1], 'step': x[2], 'result': res}), PRECEDENCE: None},
    # 20. LCM(List)
    {NAME: OPR_LCM, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'list': x[0], 'result': res}), PRECEDENCE: None},
    # 21. GCD(List)
    {NAME: OPR_GCD, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'LIST': x[0], 'result': res}), PRECEDENCE: None},
    # 22. COUNT_MULTI(List,List): function for the 13th question
    {NAME: OPR_COUNT_MULTI, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'ls2': x[1], 'result': res}), PRECEDENCE: None},
    # 23. DIGIT(int, digit: int)
    {NAME: OPR_DIGIT, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'digit': x[1], 'result': res}), PRECEDENCE: None},
    # 24. TO_INT(float)
    {NAME: OPR_TO_INT, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'result': res}), PRECEDENCE: None},
    # 25. CALL_SYMPY(List, target: str)
    {NAME: OPR_CALL_SYMPY, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'LIST': x[0], 'target': x[1], 'result': res}), PRECEDENCE: None},
    # 26. REVERSE_DIGIT(int)
    {NAME: OPR_REVERSE_DIGIT, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'X1': x[0], 'result': res}), PRECEDENCE: None},
    # 27. SEQ_TERM(List, int)
    {NAME: OPR_SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'seq': x[0], 'index': x[1], 'result': res}), PRECEDENCE: None},
    # 28. REP_SEQ_TERM(List, int)
    {NAME: OPR_REP_SEQ_TERM, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'seq': x[0], 'index': x[1], 'result': res}), PRECEDENCE: None},
    # 29. MAKE_PAIR(str, str_or_int)
    {NAME: OPR_MAKE_PAIR, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'X1': x[0], 'X2': x[1], 'result': res}), PRECEDENCE: None},
    # 30. COUNT(List) 
    {NAME: OPR_COUNT, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'result': res}), PRECEDENCE: None},
    # 31. LT(List, int)
    {NAME: OPR_LT, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'x1': x[1], 'result': res}), PRECEDENCE: None},
    # 32. LE(List, int)
    {NAME: OPR_LE, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'x1': x[1], 'result': res}), PRECEDENCE: None},
    # 33. GT(List, int)
    {NAME: OPR_GT, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'x1': x[1], 'result': res}), PRECEDENCE: None},
    # 34. GE(List, int)
    {NAME: OPR_GE, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'x1': x[1], 'result': res}), PRECEDENCE: None},
    # 35. LIST_CONCAT(ls1: List[Any], ls2: List[Any]) -> List[Any]
    {NAME: OPR_LIST_CONCAT, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'ls2': x[1], 'result': res}), PRECEDENCE: None},
    # 36. LIST_INDEX(ls: List[Union[int,float,str]],item: Union[str, int, float]) -> int
    {NAME: OPR_LIST_INDEX, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'item': x[1], 'result': res}), PRECEDENCE: None},
    # 37. LIST_REPLACE(ls: List[Union[int,float,str]], n: int, item: Union[int,float,str]) -> List[Union[int,float,str]]
    {NAME: OPR_LIST_REPLACE, ARITY: 3, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'ls1': x[0], 'index': x[1], 'item': x[2], 'result': res}), PRECEDENCE: None},
    # 38. CEIL(UNION[int, float]) -> int
    {NAME: OPR_CEIL, ARITY: 1, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'x1': x[0], 'result': res}), PRECEDENCE: None},
    # 39. LIST_MUL(lst: List[Any], n: int) -> List[Any]
    {NAME: OPR_LIST_MUL, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'lst': x[0], 'n': x[1], 'result': res}), PRECEDENCE: None},
    # 40. CHANGE_DIGIT(original: int, place: int, digit: int)
    {NAME: OPR_CHANGE_DIGIT, ARITY: 3, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'original': x[0], 'place': x[1], 'change': x[2], 'result': res}), PRECEDENCE: None},
    # 40. GET_DIGIT(original: int, place: int)
    {NAME: OPR_GET_DIGIT, ARITY: 2, COMMUTATIVE: False, ISVOID: False,
     CONVERT: (lambda res, *x: {'original': x[0], 'place': x[1], 'result': res}), PRECEDENCE: None}
]

OPR_TOKENS = [x[NAME] for x in OPR_VALUES]
OPR_SZ = len(OPR_TOKENS)

# Special operator ids
OPR_NEW_EQN_ID = OPR_TOKENS.index(OPR_NEW_EQN)
OPR_DONE_ID = OPR_TOKENS.index(OPR_DONE)
OPR_EQ_SGN_ID = OPR_TOKENS.index(OPR_EQ)
OPR_PLUS_ID = OPR_TOKENS.index(OPR_ADD)

OPR_SPECIAL = OPR_TOKENS[:OPR_EQ_SGN_ID]
OPR_NON_SPECIAL = OPR_TOKENS[OPR_EQ_SGN_ID:]

OPR_MAX_ARITY = max(op[ARITY]
                    for op in OPR_VALUES)
OPR_TOP_LV = {i
              for i, token in enumerate(OPR_VALUES)
              if token[ISVOID]}
