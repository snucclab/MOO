from argparse import ArgumentParser
from pathlib import Path
import yaml
import re
import random
import josa_converter
from convert import tokenize_string
from simulate.func import *

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

def yaml_dump(filepath, data):
    with open(filepath, "w", encoding="utf-8") as file_descriptor:
        yaml.dump(data, file_descriptor, allow_unicode=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_path', '-t', type=str, default='./7-4-1.yaml')
    parser.add_argument('--samples_per_template', '-N', type=int, default=50)
    parser.add_argument('--output_path', '-o', type=str, default='./output/1-2-1_samples.yaml')
    args = parser.parse_args()

    special_pattern = re.compile('[<>]')
    num_pattern = re.compile('num\.\d+')

    count = 1
    problem_with_equations = []
    while count != args.samples_per_template+1:
        data = yaml_loader(args.template_path)
        vocab = yaml_loader('./vocab.yaml')

        problem = data['problem']
        func_call = data['function-call']

        variable_dict = {}
        if func_call is not None:
            exec('RESULT='+func_call)
            if 'arithmatic_prog' in func_call:
                # RESULT is list
                for idx, el in enumerate(RESULT):
                    variable_dict['num.'+str(idx)] = el
                    problem = re.sub(num_pattern,str(el), problem)

        for item_name, item_value in data['variable-sampling'].items():

            start_val = item_value['range'][0]
            end_val = item_value['range'][1]

            ### 추후에 고칠 예정: if문 대신 함수 (#rsk)
            if type(start_val) != int:
                start_val = re.sub(special_pattern, '', str(start_val))
                temp_key = re.findall(num_pattern, str(start_val))[0]
                start_val = start_val.replace(temp_key, str(variable_dict[temp_key]))
                start_val = eval(start_val)
            ### 추후에 고칠 예정: if문 대신 함수 (#rsk)
            if type(end_val) != int:
                end_val = re.sub(special_pattern, '', str(end_val))
                temp_key = re.findall(num_pattern, str(end_val))[0]
                end_val = end_val.replace(temp_key, str(variable_dict[temp_key]))
                end_val = eval(end_val)

            if item_value['type'] == 'int':
                variable_dict[item_name] = random.randint(item_value['range'][0], item_value['range'][1])

            if item_value['type'] == 'float':
                variable_dict[item_name] = random.randint(item_value['range'][0] * 100,
                                                          item_value['range'][1] * 100) / 100

        s_variable_dict = {key: str(value) for key, value in variable_dict.items()}
        for key, value in s_variable_dict.items():
            problem = problem.replace('<' + key + '>', value)

        problem = re.sub('[<>]', '', problem)
        p_tokens = tokenize_string(problem)

        keys = []
        for idx, token in enumerate(p_tokens):
            for key in vocab:
                if key in token:
                    keys.append(p_tokens[idx] + p_tokens[idx + 1])
        keys = list(set(keys))

        chosen_list = []
        random_dict = {}
        for idx, key in enumerate(keys):
            vocab_list = vocab.get(re.sub('\.\d+', '', key))
            val = random.choice(vocab_list)
            while True:
                if val not in chosen_list:
                    break
                val = random.choice(vocab_list)
            random_dict[key] = val
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

        equations = data['equations']

        for variable_key, variable_value in s_variable_dict.items():
            equations = equations.replace(variable_key, variable_value)

        for tokenized_key, tokenized_value in tokenized_dictionary.items():
            equations = equations.replace('<' + tokenized_key + '>', tokenized_value)

        equations = equations.replace("<", "")
        equations = equations.replace(">", "")
        equations = re.sub(r'R0: ', '', equations)
        equations = re.sub(r'R\d+: ', '\n', equations)

        result = list()
        result.append(problem)
        result.append(equations)

        count += 1

    yaml_dump(args.output_path, result)