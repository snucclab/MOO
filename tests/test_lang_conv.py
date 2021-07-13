import pytest

from common.model.const import DEF_ENCODER
from common.sys.const import MODULE_ROOT_PATH
from common.sys.convert import string_to_text_instance
from evaluate import Executor
from solver import *

_NUMBERING = re.compile('^\\d+\\.')
_OPERATION_ONLY = re.compile('^R\\d+:\\s*([^#]+)(?:#.*)?')


@pytest.fixture(scope="session")
def executor():
    executor = Executor()
    yield executor
    executor.close()


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEF_ENCODER)
    yield tokenizer


def _verify_indent(index: int, template: str):
    templates = template.split('\n')
    for lineno, line in enumerate(templates):
        if line.endswith('\n'):
            line = line[:-1]
        if line:
            assert re.fullmatch('^( {4})*[^\\s]+.*$', line), \
                '코드 들여쓰기는 반드시 공백문자 4개여야 합니다. (%s번 문제 L#%3d)\n"%s"' % (index + 1, lineno, line)
            assert '"' not in line, \
                '코드에는 쌍따옴표를 사용할 수 없습니다. (%s번 문제 L#%3d)\n"%s"' % (index + 1, lineno, line)


def _convert_and_run(index: int, moo_code: str, text: str, expected: str,
                     executor: Executor, tokenizer = None):
    text = string_to_text_instance(text, tokenizer=tokenizer)
    executions = python_code_to_executions(moo_code)
    pycode = execution_to_python_code(executions, text.word_info[0], indent=4)
    adjusted, answer = executor.run(pycode)

    assert expected == answer, \
        '정답 불일치: %s번 문제의 정답은 "%s", 하지만 계산된 답은 "%s"입니다.' % (index + 1, expected, answer)
    assert '##@@@@' not in adjusted, \
        '코드에서 사라져야 하는 부분이 사라지지 않았습니다! (%s번 문제)' % (index + 1)
    _verify_indent(index+1, adjusted)


def test_solver_conversion(executor, tokenizer):
    lang_spec_md = MODULE_ROOT_PATH / 'MOO-Language.md'
    problems = []
    code_templates = []
    answers = []

    with lang_spec_md.open('r+t') as fp:
        example_begin = False
        code_begin = False
        for line in fp.readlines():
            line = line.strip()
            if line.startswith('## Examples'):
                example_begin = True
            elif not example_begin:
                continue

            if code_begin:
                if line == '```':
                    code_begin = False
                else:
                    line = _OPERATION_ONLY.sub('\\1', line).strip()
                    code_templates[-1].append(line)
            else:
                if _NUMBERING.match(line) is not None:
                    line = _NUMBERING.sub('', line).strip()
                    problems.append(line)
                elif line.startswith('```yaml'):
                    code_begin = True
                    code_templates.append([])
                elif line.startswith('답:'):
                    answers.append(line.replace('답:', '').strip())

        assert len(problems) == len(code_templates) == len(answers)

    for i, (prob, code, expected) in enumerate(zip(problems, code_templates, answers)):
        code = '\n'.join(code)
        _convert_and_run(i, code, prob, expected, executor, tokenizer)
