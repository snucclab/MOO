from typing import List, Dict

from common.simulate.types import Problem
import random
import simulate.josa_converter
import yaml
from simulate.convert import tokenize_string


def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

class Simulator:
    def prob_gen(self, template: Dict) -> [str, str]:
        problem = template['problem']
        # print(problem)

        numbers = []
        variable_dict = {}
        for item_name, item_value in template['variable-sampling'].items():
            if item_value['type'] == 'int':
                variable_dict[item_name] = random.randint(item_value['range'][0], item_value['range'][1])

            if item_value['type'] == 'float':
                variable_dict[item_name] = random.uniform(item_value['range'][0], item_value['range'][1])
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

        # result = re.search(r"\[([A-Za-z0-9_]+)\]", problem)
        # result = problem.split('<', 1)[1].split('>')[0]
        # print(result)

        problem = problem.replace("<", "")
        problem = problem.replace(">", "")
        problem = problem.replace(".0", "")
        problem = problem.rstrip()
        # problem = str(count)+". "+problem

        problem = josa_converter.replace_josa(problem)
        # print('tokenize')

        tokenized_problem_list = tokenize_string(problem)
        print('tokenized_problem_list')
        print(tokenized_problem_list)

        tokenized_list = []
        for value, key in enumerate(tokenized_problem_list):
            tokenized_list.append([key, value])
        print('tokenized_list')
        print(tokenized_list)

        tokenized_dictionary = {key: str(value) for value, key in enumerate(tokenized_problem_list)}
        # print('enumerated tokenized_problem_list')
        # print(tokenized_dictionary)
        # print(tokenized_dictionary)

        for key, value in tokenized_list:
            if int(value) < 10:
                tokenized_list[int(value)][1] = "{}{}".format("0", value)
        for key, value in tokenized_list:
            tokenized_list[int(value)][1] = "{}{}".format("_", value)
        print(tokenized_list)

        tokenized_list_index = []
        for key, value in tokenized_list:
            tokenized_list_index.append(value)
        print('tokenized_list_index')
        print(tokenized_list_index)
        tokenized_list_index_string = ' '.join([str(token) for token in tokenized_list_index])

        for key, value in tokenized_dictionary.items():
            if int(value) < 10:
                tokenized_dictionary[key] = "{}{}".format("0", value)
        for key, value in tokenized_dictionary.items():
            tokenized_dictionary[key] = "{}{}".format("_", value)
        print(tokenized_dictionary)
        #
        # tokenized_index_list = []
        # for key, value in tokenized_dictionary.items():
        #     tokenized_index_list.append(value)
        # tokenized_index_list.sort()
        # print('tokenized_index_list')
        # print(tokenized_index_list)

        equations = template['equations']
        print(equations)

        for variable_key, variable_value in s_variable_dict.items():
            equations = equations.replace(variable_key, variable_value)

        for tokenized_key, tokenized_value in tokenized_dictionary.items():
            equations = equations.replace(tokenized_key, tokenized_value)

        equations = equations.replace("<", "")
        equations = equations.replace(">", "")
        equations = equations.replace("R0: ", "")
        equations = equations.replace("R1:", "")
        equations = equations.replace("R2:", "")
        equations = equations.replace("R3:", "")
        equations = equations.replace("R4:", "")
        equations = equations.replace("R5:", "")
        equations = equations.replace("R6:", "")
        equations = equations.replace("R7:", "")
        equations = equations.replace("R8:", "")
        equations = equations.replace("R9:", "")
        equations = equations.replace("R10:", "")
        equations = equations.replace("R11:", "")
        equations = equations.replace("R12:", "")
        equations = equations.replace("R13:", "")
        equations = equations.replace("R14:", "")
        equations = equations.replace("R15:", "")
        equations = equations.replace("R16:", "")
        equations = equations.replace("R17:", "")
        equations = equations.replace("R18:", "")
        equations = equations.replace("R19:", "")
        equations = equations.replace("R20:", "")
        equations = equations.replace("R21:", "")
        equations = equations.replace("R22:", "")
        equations = equations.replace("R23:", "")
        equations = equations.replace("R24:", "")

        print('equations')
        equations = equations.replace(" ","\n")
        #print(equations)

        # zipped_token_index = zip(*[tokenized_list_index, tokenized_problem_list])
        # print('zipped')
        # print(tokenized_list_index)
        # print(tokenized_problem_list)
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
        for template in self.templates:
            problems = []
            for idx in range(n):
                text, code_template = self.prob_gen(template)
                problems.append(Problem(text, code_template))

            results.append(problems)
        return results