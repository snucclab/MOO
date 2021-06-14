import pytest
import random
import math
import itertools
import re
import sympy
from pathlib import Path

from common.model.const import DEF_ENCODER
from common.sys.convert import string_to_text_instance
from common.solver.const import *
from solver import *
from evaluate import Executor


_ROOT_PATH = Path(__file__).parent.parent
_NUMBERING = re.compile('^\\d+\\.')
_OPERATION_ONLY = re.compile('^R\\d+:\s*([^#]+)(?:#.*)?')


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


def _convert_and_run(moo_code: str, text: str, expected: str,
                     executor: Executor, tokenizer = None):
    text = string_to_text_instance(text, tokenizer=tokenizer)
    executions = python_code_to_executions(moo_code)
    pycode = execution_to_python_code(executions, text.word_info, indent=4)
    adjusted, answer = executor.run(pycode)

    assert expected == answer


def test_solver_conversion(executor, tokenizer):
    lang_spec_md = _ROOT_PATH / 'MOO-Language.md'
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

    for prob, code, expected in zip(problems, code_templates, answers):
        code = '\n'.join(code)
        _convert_and_run(code, prob, expected, executor, tokenizer)
