import random
import re
from decimal import Decimal
from pathlib import Path
from typing import List, Dict

import yaml

import importlib
import simulate.josa_converter
from common.simulate.types import Problem
from common.sys.convert import tokenize_string


def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data


def _check_is_not_none(value):
    return value is not None


_GLOBAL = importlib.import_module('simulate.func').__dict__


class Simulator:
    vocab = []

    def prob_gen(self, template: Dict) -> [str, str]:

        problem = template['problem']
        func_call = template.get('function-call', None)

        RESULT = {}

        if _check_is_not_none(template['variable-sampling']):
            for i, (item_name, item_value) in enumerate(template['variable-sampling'].items()):
                item_range = item_value['range'].copy()

                for j, value in enumerate(item_range):
                    if type(value) is str:
                        temp_keys = re.findall(r'<\w+\.\d+>', value)
                        for key in temp_keys:
                            value = value.replace(key, RESULT[key])
                        item_range[j] = eval(value)

                if item_value['type'] == 'int':
                    sampled = random.randint(item_range[0], item_range[1] - 1)
                else:
                    sampled = Decimal(random.randint(item_range[0] * 100, item_range[1] * 100 - 1)) / 100
                RESULT['<' + item_name + '>'] = str(sampled)

        if _check_is_not_none(func_call):
            _global = _GLOBAL.copy()

            for command in func_call.split(';'):
                for key, value in RESULT.items():
                    command = command.replace(key, value)

                command = command.strip()
                command = re.sub('\\)$', ', result)', command)
                exec(command, _global, {'result': RESULT})

        for key, value in RESULT.items():
            problem = problem.replace(key, value)

        keys = sorted(set([match.group(0) for match in re.finditer('<[^>]+>', problem)]))
        chosen_list = []
        random_dict = {}
        for idx, raw_key in enumerate(keys):
            key = raw_key[1:-1]
            if _check_is_not_none(template['list-sampling']) and key in template['list-sampling']:
                vocab_list = template['list-sampling'][key]
            else:
                vocab_list = self.vocab.get(re.sub(r'\.\d+$', '', key))

            vocab_list = [item for item in vocab_list if item not in chosen_list]
            val = random.choice(vocab_list)
            random_dict[raw_key] = val
            chosen_list.append(val)

        for key, value in random_dict.items():
            RESULT[key] = value
            problem = problem.replace(key, value)

        problem = josa_converter.replace_josa(problem)
        tokenized_problem_list = tokenize_string(problem)

        tokenized_dictionary = {}
        for index, token in enumerate(tokenized_problem_list):
            if token not in tokenized_dictionary:
                tokenized_dictionary[token] = '_%s' % index

        equations = template['equations']
        equations = re.sub('R\\d+: ', '\n', equations)
        equations = re.sub('\n+', '\n', equations).strip()

        for key, value in RESULT.items():
            equations = equations.replace(key, tokenized_dictionary.get(value, value))

        # zipped_token_index = ' '.join(['_%d:%s' % (i, token) for i, token in enumerate(tokenized_problem_list)])
        return problem, equations

    def load_templates(self, path: str):
        """
        주어진 템플릿 설정의 list를 읽어 template을 등록합니다.

        :param str path: Path to load
        """
        # Read templates from templateroot
        self.templates = []
        for file in Path(path).glob('**/*.yaml'):
            with file.open('r+t', encoding='UTF-8') as fp:
                template_in_file = yaml.safe_load(fp)
                assert type(template_in_file) is dict

                template_in_file['id'] = str(file.absolute())
                self.templates.append(template_in_file)

    def load_vocab(self, path: str):
        """
        주어진 템플릿 설정의 list를 읽어 template을 등록합니다.

        :param str path: Path to load
        """
        with Path(path).open('rt', encoding='UTF-8') as fp:
            self.vocab = yaml.safe_load(fp)

    def set_seed(self, seed: int):
        """
        난수생성기의 초기값을 seed로 초기화합니다.

        :param int seed: 초기값
        """
        random.seed(seed)

    def generate(self, n: int) -> List[List[Problem]]:
        """
        각 template마다 n개의 새로운 문제를 생성합니다.

        :param int n: template 당 생성될 문제의 개수
        """
        results = []
        for idx, template in enumerate(self.templates):
            print(str(idx) + "번째 템플릿 생성중: %s" % template['id'])
            problems = []
            for idx in range(n):
                text, code_template = self.prob_gen(template)
                problems.append(Problem(template['id'], text, code_template))

            results.append(problems)
        return results
