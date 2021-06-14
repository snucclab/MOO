from typing import List, Dict, Any, Tuple
from common.solver.const import *
from common.solver.types import Execution


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
    pass


def execution_to_python_code(expression: List[Execution],
                             word_mappings: List[Dict[str, Any]], indent: int = 4) -> str:
    """
    결과로 획득한 expression을 python code로 변환합니다.

    :param List[Execution] expression:
        Execution 객체의 List입니다.
    :param List[Dict[str,Any]] word_mappings:
        텍스트의 각 단어마다 숫자 값 또는 단어를 담고 있는 list입니다.
        'is_num'(common.sys.key.IS_NUM)은 단어가 십진숫자인지의 여부 (True/False)
        'is_var'(common.sys.key.IS_VAR)는 단어가 미지수[A, B, C, ..., Z]인지의 여부
        'is_prop'(common.sys.key.IS_PROP)는 단어가 고유명사[(가), ... 정국, ...]인지의 여부
        'value'(common.sys.key.VALUE)는 단어의 불필요한 부분을 제외한 어근의 값 (string)
        'word'(common.sys.key.WORD)는 단어 자체
    :param int indent:
        Indentation에 사용할 space 개수입니다.
    :rtype: str
    :return:
        주어진 execution 순서에 맞는 Python code string입니다.
        사람이 읽을 수 있도록 '_i' 변수명은 모두 적절한 string 값 또는 변수명으로 치환됩니다.
    """
    result = ""

    for r_id, execution in enumerate(expression):
        cur_opr = OPR_VALUES[execution.function] # dict()
        cur_arg = execution.arguments # LIST[Tuple[int, int]]

        result += execute_opr(cur_opr, cur_arg)
        

def execute_opr(opr : Dict[str, Any], arg : List[Tuple[int, int]]) -> str :
    if opr[NAME] == OPR_NEW_EQN :
        pass
    elif 
