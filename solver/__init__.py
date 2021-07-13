import re
from typing import List, Dict, Any, Tuple

from common.solver.const import *
from common.solver.types import Execution
from common.sys.const import MODULE_RSC_PATH

_MOO_LANG_PATH = MODULE_RSC_PATH / 'moo_lang'


def python_code_to_executions(code_template: str) -> List[Execution]:
    """
    주어진 python code를 execution으로 변환합니다.
    python code에 주어진 _i는 문제 속 i번째 단어를 지칭합니다.
    예를 들어, `for _9 in range(int(_11), int(_13))`와 같이 표시될 수 있습니다.
    Execution으로 변환할 때에는 이를 적절한 argument type으로 변경합니다.
    또, python code는 macro를 포함할 수 있습니다.

    :param str code_template: 변환할 python code 또는 이에 준하는 코드
    :rtype: List[Execution]
    :return: 주어진 코드에 맞는 list of Execution.
    """
    expr_list = code_template.split('\n')

    # Execution 클래스로 구성된 list를 initiate
    exec_list = []

    pattern = re.compile(r'[(),]')
    for i, expr in enumerate(expr_list):
        # expr 모양새 예시: 'SUB(_5,R0)'
        # temp 모양새 예시: [SUB, _5, R0]
        temp = re.split(pattern, expr[:-1])  # exclude last letter

        # Execution 클래스인 execution이라는 object의 function 지정 
        ex_func_idx = OPR_TOKENS.index(temp[0].strip())

        # Execution 클래스인 execution이라는 object의 arguments 지정
        # 하나의 Tuple은 (값의 타입, 값의 위치)를 나타냄.
        # - (0, i)는 사전 정의된 i번째 상수값
        # - (1, i)는 문제 속 i번째 단어에 해당하는 숫자 또는 문자열 값
        # - (2, i)는 지금까지 계산한 결과값 중 i번째 결과값
        ex_args = []

        arity = OPR_VALUES[ex_func_idx][ARITY]
        temp_args = temp[1:(arity+1)]
        for arg in temp_args:
            arg_strip = arg.strip()
            if '_' in arg:
                index = arg_strip.lstrip('_')
                ex_args.append((1, int(index)))
            elif 'R' in arg:
                index = arg_strip.lstrip('R')
                ex_args.append((2, int(index)))
            else:
                ex_args.append((0, CON_TOKENS.index(arg_strip)))

        execution = Execution(ex_func_idx, ex_args)
        exec_list.append(execution)

    return exec_list


def execution_to_python_code(expression: List[Execution],
                             word_mappings: List[Dict[str, Any]], indent: int = 4) -> str:
    """
    결과로 획득한 expression 을 python code 로 변환합니다.

    :param List[Execution] expression:
        Execution 객체의 List 입니다.
    :param List[Dict[str,Any]] word_mappings:
        텍스트의 각 단어마다 숫자 값 또는 단어를 담고 있는 list 입니다.
        'is_num'(common.sys.key.IS_NUM)은 단어가 십진숫자인지의 여부 (True/False)
        'is_var'(common.sys.key.IS_VAR)는 단어가 미지수[A, B, C, ..., Z]인지의 여부
        'is_prop'(common.sys.key.IS_PROP)는 단어가 고유명사[(가), ... 정국, ...]인지의 여부
        'value'(common.sys.key.VALUE)는 단어의 불필요한 부분을 제외한 어근의 값 (string)
        'word'(common.sys.key.WORD)는 단어 자체
    :param int indent:
        Indentation 에 사용할 space 개수입니다.
    :rtype: str
    :return:
        주어진 execution 순서에 맞는 Python code string 입니다.
        사람이 읽을 수 있도록 '_i' 변수명은 모두 적절한 string 값 또는 변수명으로 치환됩니다.
    """
    result = []

    for execution in expression:
        cur_opr = OPR_VALUES[execution.function]  # dict()
        cur_arg = execution.arguments  # LIST[Tuple[int, int]]

        if cur_opr[NAME] == OPR_NEW_EQN:
            result.clear()
            continue
        elif cur_opr[NAME] == OPR_DONE:
            break

        result.append(intprt_opr(cur_opr, cur_arg, word_mappings, len(result)))

    return '\n'.join(result)


def intprt_opr(opr: Dict[str, Any], args: List[Tuple[int, int]], word_mappings: List[Dict[str, Any]],
               op_count: int) -> str:
    keys = []
    for src, idx in args:
        if src == SRC_RESULT:
            # 이전 연산 결과 참조
            keys.append('R' + str(idx))
        elif src == SRC_NUMBER:
            # 인덱스에 해당하는 숫자 값 가져오기
            word = word_mappings[idx]
            if word[IS_NUM]:
                keys.append(word[VALUE])
            else:
                keys.append('\'' + word[VALUE] + '\'')
        else:
            # arg[0]이 0인 경우, constant 상수 값
            keys.append(CON_TOKENS[idx])

    # opr 에 해당하는 template 가져와서 식 생성
    name = opr[NAME]
    code = _load_pyt(name)
    converter = OPR_VALUES[OPR_TOKENS.index(name)][CONVERT]
    return code.format(**converter("R" + str(op_count), *keys))


def _load_pyt(name: str):
    path = _MOO_LANG_PATH / (name + '.pyt')
    with path.open('r+t', encoding='UTF-8') as fp:
        lines = fp.readlines()

    return ''.join(lines)
