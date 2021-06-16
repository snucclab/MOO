class Problem:
    #: 생성되는 문제 1개의 텍스트 string
    text: str
    #: 문제가 의도한 정답
    answer: str
    #: 답을 구하기 위한 Python code template
    code_template: str
    # 문제의 i번째 단어를 변수명으로 사용하거나, 값으로 사용하는 경우, _i와 같이 참조.
    # _i가 변수명으로 사용되거나 string으로 사용되는 경우는 별도의 casting을 하지 않으며,
    # int나 float으로 사용되는 경우는 int(_i) 또는 float(_i)와 같이 작성
    # Solver part와 협의 하에, 반복되는 코드는 macro를 사용해 간결하게 만들 수 있습니다.

    def __init__(self, text: str, code_template: str):
        self.text = text
        self.code_template = code_template

        # 아래 부분은 make_dataset.py에서 자동으로 계산합니다.
        self.code = ''
        self.executed = ''
        self.execution = []
        self.answer = ''
