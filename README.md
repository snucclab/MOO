# 인공지능 그랜드 챌린지 5차 1단계 대회 Code

이 저장소는 한국 과학기술정보통신부가 주관한 "인공지능 그랜드 챌린지 5차 1단계 대회"에 제출할 소스코드를 담고 있습니다.



## 프로젝트 구조

전체 프로젝트는 팀원의 역할에 따라 4개의 부분으로 구성됩니다.

>  **참고**: 각 파트는 소스코드를 수정할 때 반드시 아래 부분에 각 파일의 역할을 기술해주세요.



1. System Integration part: 추론 시스템의 학습용 코드 및 추론 구동을 위한 코드를 포함하는 부분입니다.

   - `/learner` 디렉토리에 추론 시스템의 학습을 위한 내부 코드를 넣어둡니다.
   - `/evaluate` 디렉토리에 추론과정 구동을 위한 내부 코드를 넣어둡니다.
   - `/common` 디렉토리는 여러 부분이 공유하는 데이터를 넣어둡니다.
     - `/common/sys` 디렉토리는 System integration이 사용할 상수 값 등을 담습니다. (problem 파일 위치 등)
     - `/common/model` 디렉토리는 Model part에 관한 상수 값 등을 담습니다. (Hyperparameter key 등)
     - `/common/solver` 디렉토리는 Solver part에 관한 상수 값 등을 담습니다. (Operator, operand 정의 등)
     - `/common/simulate` 디렉토리는 Data simulator에 관한 상수 값 등을 담습니다. (YAML key 등)
   - `/requirements.txt` 파일은 PyPI requirement 정의 파일입니다.
     - **참고**: ray는 학습에만 사용하므로 requirements.txt에는 넣지 않았습니다.
   - `/main.py` 파일은 챌린지 사무국이 지정한 추론 구동을 위한 실행 파일입니다.
   - `/train_model.py` 파일은 추론 시스템의 학습을 수행하는 실행 파일입니다.
     - **참고**: Model part는 이 파일만 사용해 debugging 해도 충분합니다. (`/main.py`와 코드를 공유합니다.)
   - `/check_dataset.py` 파일은 Dataset의 정합성을 평가하는 파일입니다.
     - **참고**: Solver part는 이 파일을 사용해 debugging 합니다.
   - `/make_dataset.py` 파일은 학습 및 평가(development)에 사용할 예제를 생성하는 파일입니다.
     - **참고**: Data simulator part는 이 파일을 사용해 debugging합니다.

   

2. Model part: 추론 시스템이 사용하는 EPT 모델과 관련 코드를 정의합니다.

   - `/model` 디렉토리에 EPT 모델을 구성하는 내부 코드를 넣어둡니다.

     - **참고**: 파이썬 기본 라이브러리를 제외하면 다음 라이브러리만을 사용합니다. `torch, numpy, transformers`
     - Optimization 관련 코드는 system integration으로 이동됩니다.

   - 필요한 경우 `/common/model`을 수정할 수 있습니다.
     단, 수정할 경우 이 코드에 의존하는 System integration part에 알려야 합니다.
     (`/train_model.py`나 `/model.py`를 직접 수정하지 않습니다!)

     

3. Solver part: EPT가 출력하는 expression sequence를 python code로 변환하고, 답을 도출합니다.

   - `/solver` 디렉토리에 Solver를 구성하는 내부 코드를 넣어둡니다.
   - 필요한 경우 `/common/solver`를 수정할 수 있습니다.
     단, 수정할 경우 이 코드에 의존하는 System integration part와 Data simulator part에 알려야 합니다.
     (다른 부분의 코드를 직접 수정하지 않습니다!)

   

4. Data simulator part: 학습에 사용할 data를 simulate하는 코드를 정의합니다.

   - `/simulate` 디렉토리에 Simulator를 구성하는 내부 코드를 넣어둡니다.
   - 필요한 경우 `/common/simulate`를 수정할 수 있습니다.
     단, 수정할 경우 System integration part에 알려야 합니다.
     (`/make_dataset.py`를 직접 수정하지 않습니다!)



## API Specification

### 대전제

모든 call은 system integration part에서 발생하고, model/solver/simulator part는 서로 독립적입니다.

따라서, model이 solver를 호출하거나, simulator가 solver를 호출하는 등의 상황은 발생하지 않습니다.



### Model spec

#### Data type spec (`common.model.types`)

> 이 부분은 System Integration에서 우선 구현합니다

```python
class Text:
	#: 토큰 인덱스. 항상 [B, S], B는 Batch size, S는 토큰 길이(가변)
	tokens: torch.LongTensor
	#: 토큰 별 단어 인덱스. tokens와 길이 동일. common.model.const.PAD_ID 값은 단어가 아닌 경우.
	word_indexes: torch.LongTensor
	#: 각 단어에 관한 정보. len(word_info) == B, len(word_info[i]) == word_indexs[i].max().
	word_info: List[List[Dict[str, Any]]]
	#   'is_num'(common.sys.key.IS_NUM)은 단어가 십진숫자인지의 여부 (True/False)
	#   'is_var'(common.sys.key.IS_VAR)는 단어가 미지수[A, B, C, ..., Z]인지의 여부
	#   'is_prop'(common.sys.key.IS_PROP)는 단어가 고유명사[(가), ... 정국, ...]인지의 여부
	#   'value'(common.sys.key.VALUE)는 단어의 불필요한 부분을 제외한 어근의 값 (string)
    #   'word'(common.sys.key.WORD)는 단어 자체

class Expression:
	#: 연산자/함수명 인덱스. 항상 [B, T], B는 Batch size, T는 토큰 길이(가변). 
	operator: torch.LongTensor
	#: 피연산자의 목록. operator와 길이 동일. common.model.const.PAD_ID는 정의되지 않은 경우.
	operands: List[torch.LongTensor]
```



#### EPT class

System Integration part는 `EPT` class (`model.EPT`)에 다음 함수가 있을 것이라 가정합니다.

> **참고** 원본 EPT 코드를 사용한다면 아래 함수가 정의되어 있을 것입니다.

```python
def forward(self, text: Text, expression: Expression = None, beam: int = 3) -> dict:
    """ 
    추론을 진행합니다. 
    
    :param common.model.types.Text text:
        텍스트 관련 정보를 담고 있습니다. 필수입니다.
    :param common.model.types.Expression expression:
        훈련 상황에서는 생성할 목표 수식을 담고 있습니다. 평가 상황에서는 None으로 주어집니다.
        훈련 상황은 self.training == True를, 반대는 평가 상황을 지칭합니다.
    :param int beam:
        평가 상황에서 사용할 Beam size
    :rtype dict:
    :returns:
        다음 항목을 key로 갖는 dictionary를 반환합니다.
        - 'prediction': common.model.types.ExpressionPrediction
            훈련 상황에만 생성합니다. 입력 expression의 다음 token들의 log-확률입니다.
            오타를 방지하기 위해서 이 키 값은 common.model.key.PREDICTION으로 참조합니다.
        - 'expression': common.model.types.Expression
            평가 상황에만 생성합니다. 입력된 text에 맞는 expression으로,
            beam search의 결과로 나온 것 중 top-1만 선택합니다.
            오타를 방지하기 위해서 이 키 값은 common.model.key.EXPRESSION으로 참조합니다.
    """

def save(self, directory: str):
    """
    checkpoint 또는 저장을 수행합니다.
    model.ept.checkpoint.CheckpointingModule로부터 상속받으므로,
    별도의 변경작업은 필요치 않습니다.
    """

@classmethod
def create_or_load(self, path: str = None, **config) -> EPT:
    """
    checkpoint 또는 저장된 객체로부터 생성하기 위한 class-level method입니다.
    어떤 config가 필요한지 꼭 문서화 해주세요.
    """
```



#### Model 구현시 주의사항

1. Beam search 과정에서 operator별로 필요한 operand의 개수가 다르거나, operator별로 제한조건이 가해질 수 있습니다.
   예를 들어, operator가 숫자만 operand로 값을 취하는 경우, 이 값이 숫자인지, 또는 그 연산 결과가 숫자인지 확인해야 합니다.
   각 operator별 specification 정의는 Solver part에서 담당하며, `/common/solver` 에 저장됩니다.



### Solver Spec

#### Data type spec (`common.solver.types`)

```python
class Execution:
    #: 사용하고자 하는 함수/연산자의 이름
    function: int
    #: 함수/연산자의 인자로 들어가는 값들.
    arguments: List[Tuple[int, int]]
    # 하나의 Tuple은 (값의 타입, 값의 위치)를 나타냄.
    # - (0, i)는 사전 정의된 i번째 상수값
    # - (1, i)는 문제 속 i번째 단어에 해당하는 숫자 또는 문자열 값
    # - (2, i)는 지금까지 계산한 결과값 중 i번째 결과값
```



#### Solver package

System Integration part는 `solver` package (`solver`)에 다음 함수가 있을 것이라 가정합니다.

```python
def python_code_to_executions(code_template: str) -> List[Execution]
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

```



#### Solver 구현시 주의사항

- 변수를 생성할 때, 한국어가 들어오면 그에 맞는 로마자 표기를 사용해야 합니다. (한국어 변수명 사용 불가)
- AST의 구조를 완벽하게 따를 필요는 없지만, 최소한 AST의 부분집합과는 호환되어야 합니다.
- 파이썬 코드의 실행 부분은 System integration에서 작성하므로 작성하지 않습니다.



### Data Simulator Spec

#### Data type spec (`common.simulator.types`)

```python
class Problem:
    #: 생성되는 문제 1개의 텍스트 string
    text: str
    #: 문제가 의도한 정답
    answer: str
    #: 답을 구하기 위한 Python code moo_lang
    code_template: str
    # 문제의 i번째 단어를 변수명으로 사용하거나, 값으로 사용하는 경우, _i와 같이 참조.
    # _i가 변수명으로 사용되거나 string으로 사용되는 경우는 별도의 casting을 하지 않으며,
    # int나 float으로 사용되는 경우는 int(_i) 또는 float(_i)와 같이 작성
    # Solver part와 협의 하에, 반복되는 코드는 macro를 사용해 간결하게 만들 수 있습니다.
```



#### Simulator class

System integration part는 `Simulator` class (`simulator.Simulator`)에 다음 함수가 있을 것이라 가정합니다.

```python
def load_templates(self, templates: List[dict]):
    """
    주어진 템플릿 설정의 list를 읽어 template을 등록합니다.
    
    :param List[dict] templates: 각 template의 설정값
    """

def set_seed(self, seed: int):
    """
    난수생성기의 초기값을 seed로 초기화합니다.
    
    :param int seed: 초기값
    """
    
def generate(self, n: int) -> List[List[Problem]]:
    """
    각 template마다 n개의 새로운 문제를 생성합니다.
    
    :param int n: moo_lang 당 생성될 문제의 개수
    """
```



#### Simulator 구현 시 주의사항

- 같은 seed로 생성되는 데이터는 같은 순서로 같은 결과가 나와야 합니다.
- 생성되는 code_template은 Solver로 번역된 다음에는 실행이 가능한 코드여야 합니다.



### System Integration spec

#### main.py

```pseudocode
Load model from './weights/checkpoint' using model.EPT.create_or_load()
Move model to GPU if available
Set model as evaluation mode

Read '/home/agc2021/dataset/problemsheet.json' and store (key, text) pairs into problems
Initialize answers as dict
For each text in problems
	Transform text into common.model.types.Text instance
	Generate equation using model.forward()
	Transform equation into a list of common.solver.types.Execution
	/* The following two lines will be shared with train_model.py, check_dataset.py */
	Transform equation into python code using solver.execution_to_python_code()
	Execute python code with timeout (0.5s) and get an answer (type: string)
	Set answers[key] as {'answer': answer, 'equation': code}

Dump answers into './answersheet.json'
Finalize everything
```



#### train_model.py

```pseudocode
/* Note: actual implementation will use parallelization.*/
Read command-line arguments, including datasetpath, checkpointpath, hyperparam
Read dataset from datasetpath (i.e., generate all required data types for each problem)

Create model based on hyperparam, using model.EPT.create_or_load()
Move model to GPU if available
Set model as training mode
Set dataset as training mode

/* Training */
For i in 0...T:
	Create a list of mini-batches
	For b in mini-batches:
		Generate equation using model.forward()
		Compare equation with gold-standard and compute loss
		Update model by back-propagation

Store the model to checkpointpath, using model.save()
Store the tokenizer to checkpointpath

/* Testing */
Set dataset as testing mode
Set correct problems as 0
For item in dataset:
	/* The following two lines will be shared with train_model.py, check_dataset.py */
	Transform equation into python code using solver.execution_to_python_code()
	Execute python code with timeout (0.5s) and get an answer (type: string)
	Verify whether the answer is the same as expected one
	if same
		Add 1 to correct problems

Print or log the accuracy
```



#### check_dataset.py

```pseudocode
Read command-line arguments, including datasetpath
Read dataset from datasetpath (i.e., generate all required data types for each problem)

For item in dataset:
	Extract execution of the item
	Transform equation into a list of common.solver.types.Execution
	/* The following two lines will be shared with train_model.py, main.py */
	Transform equation into python code using solver.execution_to_python_code()
	Execute python code with timeout (0.5s) and get an answer (type: string)
	Verify whether the answer is the same as expected one
	if not same
		Report the exception
```



#### make_dataset.py

```pseudocode
Read command-line arguments, including templateroot, numitems, datasetpath, seed
Create a simulator of type simulator.Simulator
Register seed using simulator.set_seed()

Read templates from templateroot
Register templates using simulator.load_templates()

Get generated items using simulator.generate()
Store generated items into datasetpath
```



#### common.sys.Dataset spec

- Read (text, python macro, answer) from the dataset
- Transformation spec
  - text = common.model.types.Text
  - python macro = common.model.types.Expression
  - answer = str

#### `dataset.json` 및 `split`파일 spec

##### dataset.json

```javascript1.5
{
  "ID (string)": {
    "question": "문제 텍스트 (String)", 
    "answer": "정답 (String)",
    "equation": "실행해야 하는 프로그램 코드 (String)", 
    "execution": [  // Number(int)의 list의 list
      [f0, s00, a00, s01, a01, ...],  // f0는 operator index, s00는 operand0의 source index, a00은 해당 source에서 operand0의 index
      [f1, s10, a10, s11, a11, ...],
      ...
    ]
  }
}
```

* source index에 대해서는 `common.sys.key.SRC_*` 참고
* 함수 f가 argument를 N개 받는다면, execution은 2N+1개의 항목이 있어야 함
* 실제 훈련 과정에서는 equation은 사용하지 않음

##### split

* dataset.json이 위치하는 디렉터리 속 `split` 폴더에 `train`, `dev`, `test` 명칭으로 존재함
* 각 행은 이 split에 속하는 ID 값이며, 행 구분자는 `\n`.

##### docker settings
```shell
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install transformers==4.6.1 sympy==1.8
```

##### error message
- 환경구성 오류) [도커파일빌드] 도커 이미지 전송에 실패하였습니다. 잠시 후 다시 시도해주세요. 도커 이미지 전송은 종종 네트워크 환경으로 인해 타임아웃이 발생할 수 있습니다.:
- 채점 오류) invalid character 'ï' looking for beginning of value
- 이하 환경 구성 오류
- [도커파일빌드] 빌드가 실패하였습니다. 잘못된 설치 옵션 명령어를 넣었는지 다시 한번 확인해주세요: ...(중략)... n compiling Pillow from source. Please see the install instructions at: https://pillow.readthedocs.io/en/latest/installation.html ---------------------------------------- [0m [91mERROR: Command errored out with exit status 1: /home/agc2021/.pyenv/versions/3.8.3/bin/python3.8 -u -c import sys, setuptools, tokenize; sys.argv[0] = /tmp/pip-install-v0mblh86/pillow/setup.py; __file__=/tmp/pip-install-v0mblh86/pillow/setup.py;f=getattr(tokenize, open, open)(__file__);code=f.read().replace(\r\n, \n);f.close();exec(compile(code, __file__, exec)) install --record /tmp/pip-record-b6w9ejng/install-record.txt --single-version-externally-managed --compile Check the logs for full command output. [0m [91mWARNING: You are using pip version 19.2.3, however version 21.3.1 is available. You should consider upgrading via the pip install --upgrade pip command. [0m Removing intermediate container a554a2b45b60
- [도커파일생성] 프로젝트 구조가 올바르지 않습니다. 압축을 해제한 디렉토리 구조가 규정과 일치하는지 다시 한번 확인해주세요. - 디렉토리에 프로젝트 폴더가 존재하지 않습니다.