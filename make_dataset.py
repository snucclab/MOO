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
from simulate import Simulator
from yaml import dump as yaml_dump
from json import dump as json_save
from random import seed

from solver import python_code_to_executions, execution_to_python_code


def read_arguments():
    parser = ArgumentParser()

    parser.add_argument('--template', '-template', '-t', type=str, required=False, default="./resources/tot_temp",
                        help='Root directory of template YAML files')
    parser.add_argument('--vocab', '-vocab', '-v', type=str, required=False, default="./resources/vocab.yaml",
                        help='Root directory of template YAML files')
    parser.add_argument('--num-item', '--num-sample', '-item', '-sample', '-n', type=int, required=False, default=10,
                        help='Number of items generated for each template file')
    parser.add_argument('--output', '-out', '-o', type=str, required=False, default="./resources/new_dataset",
                        help='Root directory for saving output dataset files')
    parser.add_argument('--seed', '-seed', '-s', type=int, default=1029,
                        help='Random seed for generating items')
    parser.add_argument('--time-limit', '-limit', '-l', type=float, default=0.5,
                        help='Time limit for evaluating python code')
    parser.add_argument('--tokenizer', '-tokenizer', '-z', type=str, default=DEF_ENCODER,
                        help='Pre-trained Tokenizer')
    return parser.parse_args()


if __name__ == '__main__':
    # Read command-line arguments, including templateroot, numitems, datasetpath, seed
    args = read_arguments()
    environ['DEBUG'] = 'True'
    # Create a simulator of type simulator.Simulator
    simulator = Simulator()
    # Register seed using simulator.set_seed()
    simulator.set_seed(args.seed)
    seed(args.seed)
    # Create executor to check the dataset
    executor = Executor(time_limit=0.5)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Register templates using simulator.load_templates()
    simulator.load_templates(args.template)
    simulator.load_vocab(args.vocab)

    try:
        # Get generated items using simulator.generate()
        problems = {}
        splits = defaultdict(list)
        # Convert code_template to python codes
        for template in simulator.generate(args.num_item):
            print('%s 변환중' % template[0].template)
            for i, item in enumerate(template):
                text = string_to_text_instance(item.text, tokenizer)
                execution = python_code_to_executions(item.code_template)
                raw_code = execution_to_python_code(execution, text.word_info[0])
                item.code, item.executed = executor.run(raw_code)
                item.answer = item.executed
                item.execution = [x.to_list() for x in execution]

                assert ALL_KOREAN_PATTERN.match(item.code) is None, \
                    '코드에는 한글이 포함될 수 없습니다(Template %s).\n\t실행한 코드\n%s' % (item.template, item.code)
                assert '\n' not in item.executed, \
                    '답은 오직 하나여야 합니다(Template %s). 지금은 %s개의 답이 출력되었습니다: %s' % \
                    (item.template, item.executed.count('\n') + 1, item.executed.split('\n'))
                if NUMBER_PATTERN.fullmatch(item.executed):
                    assert '.' not in item.executed or UNDER_TWO_DIGIT.fullmatch(item.executed) is not None, \
                        '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(정수이거나 소숫점 이하 두자리)에 맞지 않습니다(Template %s).' % \
                        (item.executed, item.template)
                elif FRACTION_PATTERN.fullmatch(item.executed) is None:
                    assert ALL_KOREAN_PATTERN.fullmatch(item.executed) or VARIABLE_PATTERN.fullmatch(item.executed) is not None, \
                        '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(텍스트인 경우 기타 기호 없이 한글만)에 맞지 않습니다(Template %s).' % \
                        (item.executed, item.template)

                assert item.answer == item.executed, \
                    '기대한 답 "%s"(이)가 계산된 답 "%s"(와)과 일치하지 않습니다(Template %s)!\n\t문제: "%s"\n토큰화: "%s"\n\t실행한 코드\n%s' % \
                    (item.answer, item.executed, item.template, item.text, tokenizer.decode(text.tokens), item.code)

                key = str(len(problems))
                problems[key] = item
                splits['train' if i % 10 < 8 else ('dev' if i % 10 == 8 else 'test')].append(key)

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
            json_save(obj_to_write, fp, ensure_ascii=False)
        # (4) split
        for key, split in splits.items():
            with (experiments / key).open('w+t', encoding='UTF-8') as fp:
                fp.writelines([line + '\n' for line in split])
        # (5) dataset.yaml
        with (output / 'dataset.yaml').open('w+t', encoding='UTF-8') as fp:
            obj_to_write = [{QUESTION: prob.text, ANSWER: prob.answer,
                                       EQUATION: prob.code, 'id': key, 'template': prob.template}
                            for key, prob in problems.items()]
            yaml_dump(obj_to_write, fp, allow_unicode=True, default_style='|', line_break=True)
    finally:
        # Finalize the executor
        executor.close()
