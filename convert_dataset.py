from argparse import ArgumentParser
from collections import defaultdict
from os import environ
from pathlib import Path

from transformers import AutoTokenizer

from common.model.const import DEF_ENCODER
from common.sys.convert import string_to_text_instance
from common.sys.key import QUESTION, ANSWER, EQUATION, EXECUTION
from common.sys.pattern import *
from evaluate import Executor
from json import dump as json_save
from json import load as json_load

from solver import python_code_to_executions, execution_to_python_code


def read_arguments():
    parser = ArgumentParser()

    parser.add_argument('--template', '-template', '-t', type=str, required=True,
                        help='Root path of template JSON file')
    parser.add_argument('--output', '-out', '-o', type=str, required=True,
                        help='Root directory for saving output dataset files')
    parser.add_argument('--time-limit', '-limit', '-l', type=float, default=0.5,
                        help='Time limit for evaluating python code')
    parser.add_argument('--tokenizer', '-tokenizer', '-z', type=str, default=DEF_ENCODER,
                        help='Pre-trained Tokenizer')
    return parser.parse_args()


if __name__ == '__main__':
    # Read command-line arguments, including templateroot, numitems, datasetpath, seed
    args = read_arguments()
    environ['DEBUG'] = 'True'
    # Create executor to check the dataset
    executor = Executor(time_limit=0.5)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Read templates from templateroot
    templates = {}
    with Path(args.template).open('r+t', encoding='UTF-8') as fp:
        templates.update(json_load(fp))
        assert type(templates) is dict

    # Get generated items
    problems = {}
    splits = defaultdict()
    # Convert code_template to python codes
    for _, temp in templates.items():
        item = temp
        text = string_to_text_instance(item['question'], tokenizer)
        execution = python_code_to_executions(item['equation'])

        raw_code = execution_to_python_code(execution, text.word_info[0])
        item['code'], item['executed'] = executor.run(raw_code)
        item['execution'] = [x.to_list() for x in execution]
        print(item['execution'])

        assert ALL_KOREAN_PATTERN.match(item['code']) is None, \
            '코드에는 한글이 포함될 수 없습니다.\n\t실행한 코드\n%s' % item['code']
        assert '\n' not in item['executed'], \
            '답은 오직 하나여야 합니다. 지금은 %s개의 답이 출력되었습니다: %s' % \
            (item['executed'].count('\n') + 1, item['executed'].split('\n'))
        if NUMBER_PATTERN.fullmatch(item['executed']):
            assert '.' not in item['executed'] or UNDER_TWO_DIGIT.fullmatch(item['executed']) is not None, \
                '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(정수이거나 소숫점 이하 두자리)에 맞지 않습니다.' % item['executed']
        elif FRACTION_PATTERN.fullmatch(item['executed']) is None:
            assert ALL_KOREAN_PATTERN.fullmatch(item['executed']) is not None, \
                '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(텍스트인 경우 기타 기호 없이 한글만)에 맞지 않습니다.' % item['executed']

        assert item['answer'] == item['executed'], \
            '기대한 답 "%s"(이)가 계산된 답 "%s"(와)과 일치하지 않습니다!\n\t문제: "%s"\n토큰화: "%s"\n\t실행한 코드\n%s' % \
            (item['answer'], item['executed'], item['text'], tokenizer.decode(text.tokens), item['code'])

        index = len(problems)
        key = str(index)
        problems[key] = item
        split_name = 'train' if index % 10 < 8 else ('dev' if index % 10 == 8 else 'test')
        splits[split_name].append(key)

    # Store generated items into datasetpath
    output = Path(args.output)
    experiments = output / 'split'
    if not experiments.exists():
        experiments.mkdir(parents=True)
    # (1) problemsheet.json
    with (output / 'problemsheet.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {QUESTION: prob.text}
                        for key, prob in problems.items()}
        json_save(obj_to_write, fp)
    # (2) answersheet.json
    with (output / 'answersheet.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {ANSWER: prob.answer, EQUATION: prob.code}
                        for key, prob in problems.items()}
        json_save(obj_to_write, fp)
    # (3) dataset.json
    with (output / 'dataset.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {QUESTION: prob.text, ANSWER: prob.answer,
                                   EQUATION: prob.code, EXECUTION: prob.execution}
                        for key, prob in problems.items()}
        json_save(obj_to_write, fp)
    # (4) split
    for key, split in splits.items():
        with (experiments / key).open('w+t', encoding='UTF-8') as fp:
            fp.writelines([line + '\n' for line in split])

    # Finalize the executor
    executor.close()
