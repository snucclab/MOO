from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoTokenizer

from common.model.const import DEF_ENCODER
from common.sys.convert import string_to_text_instance
from common.sys.key import QUESTION, ANSWER, EQUATION, EXECUTION
from evaluate import Executor
from simulate import Simulator
from yaml import load as yaml_load
from json import dump as json_save
from random import shuffle, seed

from solver import python_code_to_executions, execution_to_python_code


def read_arguments():
    parser = ArgumentParser()

    parser.add_argument('--template', '-template', '-t', type=str, required=True,
                        help='Root directory of template YAML files')
    parser.add_argument('--num-item', '--num-sample', '-item', '-sample', '-n', type=int, required=True,
                        help='Number of items generated for each template file')
    parser.add_argument('--output', '-out', '-o', type=str, required=True,
                        help='Root directory for saving output dataset files')
    parser.add_argument('--seed', '-seed', '-s', type=int, default=9172,
                        help='Random seed for generating items')
    parser.add_argument('--time-limit', '-limit', '-l', type=float, default=0.5,
                        help='Time limit for evaluating python code')
    parser.add_argument('--tokenizer', '-tokenizer', '-z', type=str, default=DEF_ENCODER,
                        help='Pre-trained Tokenizer')
    return parser.parse_args()


if __name__ == '__main__':
    # Read command-line arguments, including templateroot, numitems, datasetpath, seed
    args = read_arguments()
    # Create a simulator of type simulator.Simulator
    simulator = Simulator()
    # Register seed using simulator.set_seed()
    simulator.set_seed(args.seed)
    seed(args.seed)
    # Create executor to check the dataset
    executor = Executor(time_limit=0.5)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Read templates from templateroot
    templates = []
    for file in Path(args.template).glob('**/*.yaml'):
        with file.open('r+t', encoding='UTF-8') as fp:
            template_in_file = yaml_load(fp)
            assert type(template_in_file) is list

            templates += template_in_file

    # Register templates using simulator.load_templates()
    simulator.load_templates(templates)

    # Get generated items using simulator.generate()
    problems = []
    # Convert code_template to python codes
    for template in simulator.generate(args.num_item):
        for item in template:
            text = string_to_text_instance(item.text, tokenizer)
            item.execution = python_code_to_executions(item.code_template)
            item.code = execution_to_python_code(item.execution, text.word_info[0])
            item.executed = executor.run(item.code)

            assert item.answer == item.executed, \
                '기대한 답 "%s"(이)가 계산된 답 "%s"(와)과 일치하지 않습니다!\n\t문제: "%s"\n토큰화: "%s"\n\t실행한 코드\n%s' % \
                (item.answer, item.executed, item.text, tokenizer.decode(text.tokens), item.code)
            problems.append(item)

    # Shuffle the dataset
    shuffle(problems)

    # Store generated items into datasetpath
    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True)
    # (1) problemsheet.json
    with (output / 'problemsheet.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {QUESTION: prob.text}
                        for key, prob in enumerate(problems)}
        json_save(obj_to_write, fp)
    # (2) answersheet.json
    with (output / 'answersheet.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {ANSWER: prob.answer, EQUATION: prob.code}
                        for key, prob in enumerate(problems)}
        json_save(obj_to_write, fp)
    # (3) dataset.json
    with (output / 'dataset.json').open('w+t', encoding='UTF-8') as fp:
        obj_to_write = {str(key): {QUESTION: prob.text, ANSWER: prob.answer,
                                   EQUATION: prob.code, EXECUTION: prob.execution}
                        for key, prob in enumerate(problems)}
        json_save(obj_to_write, fp)
