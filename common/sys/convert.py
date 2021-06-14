from typing import List

import torch

from .pattern import *
from .key import *
from common.model.types import Text, Expression
from common.solver.types import Execution
from common.solver.const import CON_MAX, OPR_VALUES
from ..model.const import PAD_ID


def _remove_special_prefix(token: str) -> str:
    # Handle different kind of spacing prefixes...
    from transformers import SPIECE_UNDERLINE
    if token == SPIECE_UNDERLINE:
        return ' '
    if token.startswith(SPIECE_UNDERLINE):
        return token[len(SPIECE_UNDERLINE):]
    if token.startswith('##'):
        return token[2:]
    return token


def tokenize_string(text: str) -> List[str]:
    # 숫자, 변수, 고유명사 및 연산자 앞뒤로 space 추가
    text = PROPERNOUN_PATTERN.sub(' \\1 ', text)
    text = SPACING_PATTERN.sub(' \\1 ', text)

    # Space 여러개인 경우 하나로 통일
    text = re.sub('\\s+', ' ', text.strip())

    return text.split(' ')


def string_to_text_instance(text: str, tokenizer) -> Text:
    # 단어 목록 구성
    words = tokenize_string(text)
    text = ' '.join(words)

    word_info = []
    for word in words:
        number = NUMBER_BEGIN_PATTERN.fullmatch(word)
        variable = VARIABLE_BEGIN_PATTERN.fullmatch(word)
        proper = PROPER_BEGIN_PATTERN.fullmatch(word)

        if number is not None:
            value = number.group(1).replace(',', '')
        elif variable is not None:
            value = variable.group(1)
        else:
            value = proper.group(1) if proper is not None else word
            value = PAREN_PATTERN.sub(value, '')

        word_info.append({
            IS_NUM: number is not None,
            IS_VAR: variable is not None,
            IS_PROP: proper is not None,
            VALUE: value,
            WORD: word
        })

    # 토큰화
    text_encoded = tokenizer.encode(text, truncation=True)
    text_tokens = tokenizer.convert_ids_to_tokens(text_encoded)

    # 단어 위치 구성
    word_indexes = []
    word_counter = -1
    string_left = ' ' + text
    for token in text_tokens:
        if token in tokenizer.all_special_tokens:
            word_indexes.append(PAD_ID)
        else:
            # Find whether this is the beginning of the word.
            # We don't use SPIECE_UNDERLINE or ## because ELECTRA separates comma or decimal point...
            if string_left[0].isspace():
                word_counter += 1
                string_left = string_left[1:]

            token_string = _remove_special_prefix(token)
            assert string_left.startswith(token_string)
            string_left = string_left[len(token_string):]

            word_indexes.append(word_counter)

    # 단어 개수 일치여부 확인
    assert max(word_indexes) == len(word_info) - 1, \
        'Max index  %s != Word count %s' % (max(word_indexes), len(word_info))

    # Text kwargs 반환
    return Text(tokens=torch.tensor([text_encoded], dtype=torch.long),
                word_indexes=torch.tensor([word_indexes], dtype=torch.long),
                word_info=[word_info])


def equation_to_execution(equation: Expression, batch_index: int = 0, word_size: int = 0) -> List[Execution]:
    # 숫자 값 읽기
    operator = equation.operator[batch_index].tolist()
    operands = [o[batch_index].tolist() for o in equation.operands]

    executions = []
    for t, f_t in enumerate(operator):
        a_t = []
        arity = OPR_VALUES[f_t][ARITY]

        for o in operands[:arity]:
            if o < CON_MAX:
                a_t.append((SRC_CONSTANT, o))
                continue

            o -= CON_MAX
            if o < word_size:
                a_t.append((SRC_NUMBER, o))
                continue

            o -= word_size
            a_t.append((SRC_RESULT, o))

        executions.append(Execution(f_t, a_t))

    return executions
