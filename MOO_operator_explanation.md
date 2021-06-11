# AI 챌린지 연산자 설명서

1. NEW_EQN()

2. NEW_VAR()

3. DONE()

4. EQ

```python
EQ(a: float, b: float) -> bool
1) 목적/변화: 입력받은 두 인자가 같은지 비교
2) 인수값: 비교하려는 두 값
3) 반환값: 같으면 true, 다르면 false
4) 예시: 
R0: EQ(2, 2)
```

5. +

```python
+(a: float, b: float) -> float
1) 목적/변화: 입력받은 두 값을 더해서 반환
2) 인수값: 더하려는 두 값
3) 반환값: a+b, 두 인자를 더한 값
4) 예시: 
R0: +(3, 4)
```

6. -

```python
-(a: float, b: float) -> float
1) 목적/변화: 입력받은 두 값의 차를 반환
2) 인수값: 차이를 알고 싶은 두 값
3) 반환값: a-b, 두 인자의 차
4) 예시: 
R0: -(4, 3)
```

7. *

```python
*(a: float, b: float) -> float
1) 목적/변화: 입력받은 두 값의 곱을 반환
2) 인수값: 곱하고 싶은 두 값
3) 반환값: a*b, 두 인자를 곱한 값
4) 예시: 
R0: *(2, 3)
```

8. /

```python
/(a: float, b: float) -> float
1) 목적/변화: 입력 받은 두 개의 인자 중 a(첫 번째 인자)를 b(두 번째 인자)로 나눠준다
2) 인수값: 나눠주고 싶은 두 값
3) 반환값: a/b
4) 예시: 
R0: /(8,4)
```

9. ^

```python
^(a: float, b: float) -> float
1) 목적/변화: 입력 받은 두 개의 인자 중 a(첫 번째 인자)를 b(두 번째 인자)제곱을 구해준다
2) 인수값: base 값과 지수
3) 반환값: a^b (a의 b제곱)
4) 예시: 
R0: ^(2, 4)
```

10. PRINT

```python
PRINT(data : Union[float, int, str])
1) 목적/변화: 입력받은 값 출력
2) 인수값: 출력하고 싶은 float, int 또는 str
3) 반환값: 없음
4) 예시:
R0: -(10,5)
R1: PRINT(R0)
```

11. SUM

```python
SUM(ls: List[int]) -> int
1) 목적/변화: 리스트 속 정수들을 모두 더해줌
2) 인수값: 정수로 이루어진 리스트
3) 반환값: 리스트의 원소들을 모두 더한 값
4) 예시: 
  R0: LIST()
  R1: APPEND(R0, 1) # APPEND(13.) 참고
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: SUM(ls)       
```

12. LIST

```python
LIST()  
1) 목적/변화: 리스트 객체를 만든다
2) 인수값: 없음
3) 반환값: 리스트 객체
4) 예시:
	R0: LIST()
```

13. APPEND

```python
APPEND(ls: List[Any], x: Union[str,int,float,List[...]) -> List[Any]
1) 목적/변화: 리스트 속에 정수, 실수, 문자열을 추가함. 리스트 속에 리스트가 들어갈 수 있음.
2) 인수값: 정수, 실수, 문자열, 다른 리스트
3) 반환값: 정수, 실수, 문자열, 다른 리스트가 들어있는 리스트
4) 예시:
	4-1) 정수형 리스트 만들기
		R0: LIST()
		R1: APPEND(R0, 3)
		R2: APPEND(R1, 4)
	4-2) 실수형 리스트 만들기
		R0: LIST()
		R1: APPEND(R0, 3.5)
		R2: APPEND(R1, 4.5)
	4-3) 문자형 리스트 만들기
		R0: LIST()
		R1: APPEND(R0, '(가)')
		R2: APPEND(R1, '(나)')
```

14. COMB

```python
COMB(n: int, k:int) -> int
1) 목적/변화: n개 중 k개를 선택할 수 있는 경우의 수를 구한다(조합)
2) 인수값: 총 개수 n(정수), 선택할 개수 k(정수)
3) 반환값: 경우의 수
4) 예시: 
	R0: LIST()
	R1: APPEND(R0,'빨간')
	R2: APPEND(R1,'파랑')
	R3: APPEND(R2,'노랑')
	R4: COUNT(R0) # COUNT함수(29.) 참고
	R5: COMB(R4, 2)  (=3)
5) 제한 조건: n >= k (선택 할 수 있는 총 개수보다 선택할 수가 더 클 수 없다)
```

15. PERM

```python
PERM(n: int, k:int) -> int
1) 목적/변화: n개 중 k개를 선택하여 나열할 수 있는 경우의 수를 구한다(순열)
2) 인수값: 총 개수 n(정수), 선택할 개수 k(정수)
3) 반환값: 경우의 수
4) 예시: 
	R0: LIST()
	R1: APPEND(R0,'빨간')
	R2: APPEND(R1,'파랑')
	R3: APPEND(R2,'노랑')
	R4: COUNT(R0) # COUNT함수(29.) 참고
	R5: PERM(R4, 2)    (=6)
5) 제한 조건: n >= k 
```

16. MIN

```python
MIN(ls: List[Union[int,float]]) -> Union[int,float]
1) 목적/변화: 정수나 실수로 이루어진 리스트에서 가장 작은 값을 알려준다
2) 인수값: 정수나 실수로 이루어진 리스트
3) 반환값:
	정수 리스트인 경우: 정수
	실수 리스트인 경우: 실수
4) 예시:
	R0: LIST()
	R1: APPEND(R0, 4)
	R2: APPEND(R1, 5)
	R3: APPEND(R2, 8)
	R4: MIN(R3)    (=4)
5) 제한 조건: 문자열 리스트에는 사용할 수 없는 함수이다
```

17. MAX

```python
MAX(ls: List[Union[int,float]]) -> Union[int,float]
1) 목적/변화: 정수나 실수로 이루어진 리스트에서 가장 큰 값을 알려준다
2) 인수값: 정수나 실수로 이루어진 리스트
3) 반환값:
	정수 리스트인 경우: 정수
	실수 리스트인 경우: 실수
4) 예시:
	R0: LIST()
	R1: APPEND(R0, 4)
	R2: APPEND(R1, 5)
	R3: APPEND(R2, 8)
	R4: MAX(R3)    (=8)
5) 제한 조건: 문자열 리스트에는 사용할 수 없는 함수이다
```

18. RANGE

```python
RANGE(start: int, end: int, step: int) -> List[int] 
1) 목적/변화: 시작하는 숫자(start)부터 끝나는 숫자보다 1이 더 큰 숫자(end)까지
정수로 이루어진 리스트를 만든다. 리스트 속의 숫자들은 step의 값만큼 차이가 난다.
리스트 속에 있는 정수들은 크기 순서대로 배열된다. (작은 수 -> 큰 수)
2) 인수값: 시작값, 끝값, 숫자 간 차이
3) 반환값: 정수로 구성된 리스트
4) 예시:
	4-1) 1부터 10까지 나열:
		R0: RANGE(1,11,1)
	4-2) 1부터 10까지 짝수만 나열:
		R0: RANGE(2,11,2)
	4-3) 1부터 9까지 홀수만 나열:
		R0: RANGE(1,10,2)
5) 제한 조건: 시작값과 끝값도 마찬가지만, step값도 무조건 입력해야 한다.
```

19. LCM

```python
LCM(ls: List[int]) -> int
1) 목적/변화: 리스트의 정수들의 최소공배수다를 구한다
2) 인수값: 정수로 구성된 리스트
3) 반환값: 최소공배수
4) 예시:
	R0: LIST()
	R1: APPEND(R0, 5)
	R2: APPEND(R1, 8)
	R3: APPEND(R2, 9)
	R4: LCM(R3)    (=360)
5) 제한 조건: 문자열, 실수형 리스트에는 사용할 수 없는 함수이다
```

20. GCD

```python
GCD(ls: List[int]) -> int
1) 목적/변화: 리스트의 정수들의 최대공약수를 구한다
2) 인수값: 정수로 구성된 리스트
3) 반환값: 최대공약수
4) 예시:
	R0: LIST()
	R1: APPEND(R0, 50)
	R2: APPEND(R1, 80)
	R3: APPEND(R2, 90)
	R4: GCD(R3)    (=10)
5) 제한 조건: 문자열, 실수형 리스트에는 사용할 수 없는 함수이다
```

21. COUNT_MULTI

```python
COUNT_MULTI(ls1: List[int], ls2: List[int]) ->  int
1) 목적/변화: 13번 문제를 풀기 위한 특수 함수.. RANGE를 입력하면(ls1) 그 RANGE
속에 있는 숫자 중 ls2에 있는 숫자들 모두 나눠 떨어지는 숫자들의 개수를 출력한다
2) 인수값:
ls1: 찾고자 하는 RANGE
ls2: 나누어 떨어졌으면 하는 숫자들
3) 반환값: 개수
4) 예시:
	R0: RANGE(100,1000,1)
	R1: LIST()
	R2: APPEND(R1, 2)
	R3: APPEND(R2, 3)
	R4: APPEND(R2, 9)
	R5: COUNT_MULTI(R0, R4)
5) 제한 조건: ls1은 문제에서 제시하는 자리수 내에서 찾아야 하므로 다음과 같이 설정해야 한다.
한 자리수: RANGE(1,10,1)
두 자리수: RANGE(10,100,1)
세 자리수: RANGE(100,1000,1)
네 자리수: RANGE(1000,10000,1)
등등
```

22. DIGIT

```python
DIGIT(x: int, digit: int) -> int 
1) 목적/변화: 한 자리수 정수(x)가 주어지면 그 정수를 원하는 자리수의 숫자로 변환한다.
이는 그 정수에 필요한 개수(digit-1)의 0을 붙여서 생성한다.
2) 인수값: 한 자리수 정수, 
3) 반환값: 원하는 자리수의 정수
4) 예시:
	R0: DIGIT(7, 5)     (=70000)
	R1: DIGIT(2, 2)     (=20)
	R2: DIGIT(1, 3)     (=100)
	R3: DIGIT(8, 1)     (=8)
5) 제한 조건: x는 한 자리수의 정수다,(1~9) 절대로 다른 정수를 입력값으로 넣지 않는다.
```

23. TO_INT

```python
TO_INT(x: float) -> int
1) 목적/변화: 실수 x를 정수로 변환한다.
2) 인수값: 원본 실수
3) 반환값: 변환된 정수
4) 예시: 
 R0: TO_INT(3.0) = 3
```

24. CALL_SYMPY

```python
CALL_SYMPY(prob: List, target: Any) -> Union[int, float] 
1) 목적/변화: Sympy에 prob들이 들어있는 을 넘겨서 문제를 해결한다.
2) 인수값: 
	prob: 문제 str들이 들어있는 list
  target: 구하고 싶은 인자 str
3) 반환값: 정답 결과가 들어있는 int or float
4) 예시: (문제 16)
 R0: LIST() 
 R1: APPEND(R0, "A+B=30")
 R2: APPEND(R1, "A=B+B+B+B")
 R3: CALL_SYMPY(R2, 'A')
```

25. REVERSE_DIGIT

```python
REVERSE_DIGIT(x: int) -> int
1) 목적/변화: x의 각 자리수를 역순으로 변환/배열한다.
2) 인자값: 원본 정수
3) 반환값: 자리수가 역순으로 재배열 된 정수
4) 예시: 
  R0: REVERSE_DIGIT(345)      (= 543)
```

26. SEQ_TERM

```python
SEQ_TERM(sequence: List, n: int) -> Union[int, float]
1) 목적/변화: 등차수열또는 계차수열을 입력받아 n번째 인덱스에 해당하는 값을 알아내는 함수
2) 인수값:
	sequence : 수열의 리스트
  n : 몇 번째 값을 가져올 지
3) 반환값: int 또는 float으로 된 n번째 값
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: SEQ_TERM(R3, 10)
```

27. REP_SEQ_TERM

```python
REP_SEQ_TERM(List, int) -> Union[int, float]
1) 목적/변화: 등차수열또는 계차수열을 입력받아 n번째 인덱스에 해당하는 값을 알아내는 함수
2) 인수값:
  sequence : 수열의 리스트
  n : 몇 번째 값을 가져올 지
3) 반환값: int 또는 float으로 된 n번째 값
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 1)
  R5: APPEND(R4, 2)
  R6: APPEND(R5, 3)
  R7: REP_SEQ_TERM(R6, 10)
```

28. MAKE_PAIR

```python
MAKE_PAIR(str, Union[str, int]) -> List
1) 목적/변화: string 값에 대해 값을 쌍으로 묶어준다
2) 인수값: string과 int의 쌍 또는 string과 string의 쌍
3) 반환값: list
4) 예시:
  4-1) string과 int pair
  R0: MAKE_PAIR("윤기", 4)
  4-2) string과 string pair
  R0: MAKE_PAIR("가", "나")
```

29. COUNT

```python
COUNT(ls: List) -> int
1) 목적/변화: ls의 인자 개수를 세어주는 함수
2) 인수값: list
3) 반환값: list에서의 인자 개수
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: COUNT(R3)           (= 3)
```

30. LT

```python
LT(ls : List, n: int)
1) 목적/변화: 
  리스트의 각 원소들을 입력받은 인자인 n과 비교하여 n보다 작으면 1,
  n보다 크거나 같으면 0으로 바꿔준다.
2) 인수값: 비교하고 싶은 list, 기준이 되는 int값
3) 반환값: 결과 list
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: APPEND(R5, 6)
  R7: LT(R6, 4)  (= [1,1,1,0,0,0])
```

31. LE

```python
LE(List, int)
1) 목적/변화:
  리스트의 각 원소들을 입력받은 인자인 n과 비교하여 n보다 작거나 같으면 1,
  n보다 크면 0으로 바꿔준다.
2) 인수값: 비교하고 싶은 list, 기준이 되는 int값
3) 반환값: 결과 list
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: APPEND(R5, 6)
  R7: LE(R6, 4)  (= [1,1,1,1,0,0])
```

32. GT

```python
GT(List, int)
1) 목적/변화:
  리스트의 각 원소들을 입력받은 인자인 n과 비교하여 n보다 크면 1,
  n보다 작거나 같으면 0으로 바꿔준다.
2) 인수값: 비교하고 싶은 list, 기준이 되는 int값
3) 반환값: 결과 list
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: APPEND(R5, 6)
  R7: GT(R6, 4)  (= [0,0,0,0,1,1])
```

33. GE

```python
GE(List, int)
1) 목적/변화:
  리스트의 각 원소들을 입력받은 인자인 n과 비교하여 n보다 크거나 같으면 1,
  n보다 작으면 0으로 바꿔준다
2) 인수값: 비교하고 싶은 list, 기준이 되는 int값
3) 반환값: 결과 list
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: APPEND(R5, 6)
  R7: GE(R6, 4)      (= [0,0,0,1,1,1])
```

34. LIST_CONCAT

```python
LIST_CONCAT(ls1: List[Any], ls2: List[Any]) -> List[Any]
1) 목적/변화: 리스트 두 개를 붙여서 새로운 리스트를 만든다
2) 인수값: 리스트2개
3) 반환값: 리스트1개 [ls1, ls2]
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: LIST()
  R5: APPEND(R4, 4)
  R6: APPEND(R5, 5)
  R7: APPEND(R6, 6)
  R8: LIST_CONCAT(R3,R7)
5) 제한 조건: 리스트는 꼭 원하는 순서에 맞춰서 인수로 줘야한다
```

35. LIST_INDEX

```python
LIST_INDEX(ls: List[Union[int,float,str]],item: Union[str, int, float]) -> int
1) 목적/변화: 리스트와 원소가 주어지면 해당 원소의 인덱스를 반환한다
2) 인수값: 리스트, str 또는 int 또는 float 원소
3) 반환값: int
4) 예시:
  R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
  R4: APPEND(R3, 4)
  R5: APPEND(R4, 5)
  R6: APPEND(R5, 6)
	R7: LIST_INDEX(R6, 3)
```

36. LIST_REPLACE

```python
LIST_REPLACE(ls: List[Union[int,float,str]], n: int, item: Union[int,float,str]) -> List[Union[int,float,str]]
1) 목적/변화: 리스트 속에 n번째 인덱스에 해당하는 값을 item으로 바꾼다.
2) 인수값: 리스트, 바꾸고자 하는 item, 바꾸고자 하는 위치
3) 반환값: 새로 바뀐 리스트
4) 예시:
	R0: LIST()
  R1: APPEND(R0, 1)
  R2: APPEND(R1, 2)
  R3: APPEND(R2, 3)
	R4: LIST_REPLACE(R3, 2, 4)        (=[1,2,4])
5) 제한 조건: 리스트 안의 항목은 타입이 모두 통일되어야 한다.
(정수형이면 정수만, 실수형이면 실수만, 문자열이면 문자열)
```