from typing import List, Dict

from common.simulate.types import Problem
import random
import simulate.josa_converter
import yaml
from simulate.convert import tokenize_string
import re

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

class Simulator:
    def prob_gen(self, template: Dict) -> [str, str]:
        problem = template['problem']

        numbers = []
        variable_dict = {}
        for item_name, item_value in template['variable-sampling'].items():
            if item_value['type'] == 'int':
                variable_dict[item_name] = random.randint(item_value['range'][0], item_value['range'][1])

            if item_value['type'] == 'float':
                variable_dict[item_name] = random.randint(item_value['range'][0]*100, item_value['range'][1]*100)/100
        s_variable_dict = {key: str(value) for key, value in variable_dict.items()}
        for key, value in s_variable_dict.items():
            problem = problem.replace('<' + key + '>', value)

        keys = []
        for key in self.vocab:
            if key in problem:
                keys.append(key)
            else:
                pass

        random_dict = {}
        for key in keys:
            random_dict[key] = random.choice(self.vocab.get(key))

        for key, value in random_dict.items():
            problem = problem.replace(key, value)

        problem = problem.replace("<", "")
        problem = problem.replace(">", "")
        problem = problem.replace(".0", "")
        problem = problem.rstrip()

        problem = josa_converter.replace_josa(problem)
        tokenized_problem_list = tokenize_string(problem)

        tokenized_list = []
        for value, key in enumerate(tokenized_problem_list):
            tokenized_list.append([key, value])

        tokenized_dictionary = {key: str(value) for value, key in enumerate(tokenized_problem_list)}

        for key, value in tokenized_list:
            if int(value) < 10:
                tokenized_list[int(value)][1] = "{}{}".format("0", value)
        for key, value in tokenized_list:
            tokenized_list[int(value)][1] = "{}{}".format("_", value)

        tokenized_list_index = []
        for key, value in tokenized_list:
            tokenized_list_index.append(value)

        for key, value in tokenized_dictionary.items():
            if int(value) < 10:
                tokenized_dictionary[key] = "{}{}".format("0", value)
        for key, value in tokenized_dictionary.items():
            tokenized_dictionary[key] = "{}{}".format("_", value)

        equations = template['equations']
        print(problem)
        print(equations)

        return problem, equations

    def load_templates(self, templates: List[dict]):
        """
        주어진 템플릿 설정의 list를 읽어 template을 등록합니다.

        :param List[dict] templates: 각 template의 설정값
        """
        self.templates = templates

    def load_vocab(self, vocab : Dict):
        """
        주어진 템플릿 설정의 list를 읽어 template을 등록합니다.

        :param List[dict] templates: 각 template의 설정값
        """
        self.vocab = vocab

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
            print(str(idx)+"번째 템플릿 생성중")
            print(template)
            problems = []
            for idx in range(n):
                text, code_template = self.prob_gen(template)
                if text.find('14.12') != -1 :
                    print(text)
                problems.append(Problem(text, code_template))

            results.append(problems)
        return results