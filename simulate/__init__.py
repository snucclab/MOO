from typing import List, Dict

from common.simulate.types import Problem
import random
import simulate.josa_converter
import yaml
from common.sys.convert import tokenize_string
import re
import copy

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

class Simulator:
    def prob_gen(self, template: Dict) -> [str, str]:
        problem = template['problem']

        # num 끼워넣기
        numbers = []
        variable_dict = {}

        for i, item in enumerate(copy.deepcopy(template['variable-sampling']).items()):
            item_name = item[0]
            item_value = item[1]
            if i != 0:
                for j, value in enumerate(item_value['range']):
                    if type(value) is str :
                        temp_keys = re.findall(r'<num\.\d+>',value)
                        for key in temp_keys:
                            value = value.replace(key, str(variable_dict[key]))
                        item_value['range'][j] = eval(value)

            if item_value['type'] == 'int':
                variable_dict['<'+item_name+'>'] = random.randint(item_value['range'][0], item_value['range'][1]-1)
            if item_value['type'] == 'float':
                variable_dict['<'+item_name+'>'] = random.randint(item_value['range'][0]*100, item_value['range'][1]*100 -1)/100
        s_variable_dict = {key: str(value) for key, value in variable_dict.items()}

        for key, value in s_variable_dict.items():
            problem = problem.replace(key, value)

        p_tokens = tokenize_string(problem)

        keys = []
        for idx, token in enumerate(p_tokens):
            for vocab in self.vocab:
                if vocab == re.sub("[<.]","",token):
                    keys.append(p_tokens[idx] + p_tokens[idx + 1] + '>')
        keys = list(set(keys))

        chosen_list = []
        random_dict = {}
        for idx, raw_key in enumerate(keys):
            vocab = []
            key = re.sub(r'[<>]', '', raw_key)
            if key in template['list-sampling']:
                vocab_list = template['list-sampling'][key]
            else :
                vocab_list = self.vocab.get(re.sub(r'[<\.\d>]', '', raw_key))
            val = random.choice(vocab_list)
            while True:
                if val not in chosen_list:
                    break
                val = random.choice(vocab_list)
            random_dict[raw_key] = val
            chosen_list.append(val)

        for key, value in random_dict.items():
            problem = problem.replace(key, value)

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

        equations = re.sub(r'R0: ', '', equations)
        equations = re.sub(r'R\d+: ', '\n', equations)

        for key, value in s_variable_dict.items():
            if value in tokenized_dictionary:
                equations = equations.replace(key, tokenized_dictionary[value])
            else :
                equations = equations.replace(key, value)

        # for tokenized_key, tokenized_value in sorted(list(tokenized_dictionary.items()), key=lambda x: len(x[0]), reverse=True):
        #     #equations = equations.replace(tokenized_key, tokenized_value)
        #     equations = re.sub('([(,])'+ tokenized_key, r'\1'+tokenized_value, equations)


        #print(problem)
        #print(equations)

        tokenized_problem_list_endspaced = tokenized_problem_list
        for i, item in enumerate(tokenized_problem_list):
            tokenized_problem_list_endspaced[i] = "{}{}{}".format(":", item, " ")
        zipped_token_index = ''.join(
            [str(a) + b for a, b in zip(tokenized_list_index, tokenized_problem_list_endspaced)])

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
            problems = []
            for idx in range(n):
                text, code_template = self.prob_gen(template)
                problems.append(Problem(text, code_template))

            results.append(problems)
        return results