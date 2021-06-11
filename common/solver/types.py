from typing import List, Tuple
from common.sys.key import SRC_LIST, ARITY
from common.solver.const import OPR_SZ, OPR_VALUES


class Execution:
    #: 사용하고자 하는 함수/연산자의 이름
    function: int
    #: 함수/연산자의 인자로 들어가는 값들.
    arguments: List[Tuple[int, int]]
    # 하나의 Tuple은 (값의 타입, 값의 위치)를 나타냄.
    # - (0, i)는 사전 정의된 i번째 상수값
    # - (1, i)는 문제 속 i번째 단어에 해당하는 숫자 또는 문자열 값
    # - (2, i)는 지금까지 계산한 결과값 중 i번째 결과값

    def __init__(self, function: int, arguments: List[Tuple[int, int]]):
        assert function < OPR_SZ
        assert len(arguments) == OPR_VALUES[function][ARITY]
        assert all(t < len(SRC_LIST) for t, _ in arguments)

        self.function = function
        self.arguments = arguments

    def to_list(self) -> List[int]:
        return [self.function] + list(sum(self.arguments, tuple()))

    @classmethod
    def from_list(cls, items: list) -> 'Execution':
        return cls(function=items[0],
                   arguments=[tuple(items[i:i+2]) for i in range(1, len(items), 2)])
