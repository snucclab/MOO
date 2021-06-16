from argparse import ArgumentParser
from pathlib import Path
import yaml
import re
import random
import josa_converter
from convert import tokenize_string

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

def yaml_dump(filepath, data):
    with open(filepath, "w", encoding="utf-8") as file_descriptor:
        yaml.dump(data, file_descriptor, allow_unicode=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_path', '-t', type=str, default='./1-2-1.yaml')
    parser.add_argument('--samples_per_template', '-N', type=int, default=50)
    parser.add_argument('--output_path', '-o', type=str, default='./output/1-2-1_samples.yaml')
    args = parser.parse_args()

    count = 1
    problem_with_equations = []
    while count != args.samples_per_template+1:
        data = yaml_loader(args.template_path)
        # print(data)
        problem = data.get('problem')
        # print(problem)
        problem = re.sub('[<>]', '', problem)
        # print(tokenize_string(problem))
        p_tokens = tokenize_string(problem)
        vocabpath = "./vocab.yaml"
        vocab = yaml_loader(vocabpath)
        keys = []
        for idx, token in enumerate(p_tokens):
            for key in vocab:
                if key in token:
                    keys.append(p_tokens[idx]+p_tokens[idx+1])
        keys = list(set(keys))
        # print(keys)

        # num_key_appearance = []
        # for key in keys:
        #     num_key_appearance.append(key+[.])
        # print('num_key_appearance')
        # print(num_key_appearance)

        chosen_list = list()
        random_dict = {}
        for idx, key in enumerate(keys):
            # print(re.sub('\.\d+', '', key))
            vocab_list = vocab.get(re.sub('\.\d+', '', key))
            val =random.choice(vocab_list)
            while True:
                if val not in chosen_list:
                    break
                val = random.choice(vocab_list)
            random_dict[key]=val
            chosen_list.append(val)

        # print(random_dict)

        for key, value in random_dict.items():
            problem = problem.replace(key, value)
        print(problem)

        # result = re.search(r"\[([A-Za-z0-9_]+)\]", problem)
        # result = problem.split('<', 1)[1].split('>')[0]
        # print(result)

        numbers = []
        variable_dict = {}
        for item_name, item_value in data['variable-sampling'].items():

            bad_pattern = re.compile('[<>]')
            num_pattern = re.compile('num\.\d+')
            start_val = item_value['range'][0]
            end_val = item_value['range'][1]

            if type(start_val) != int:
                start_val = re.sub(bad_pattern, '', str(start_val))
                temp_key = re.findall(num_pattern, str(start_val))[0]
                start_val = start_val.replace(temp_key, str(variable_dict[temp_key]))
                start_val = eval(start_val)

            if type(end_val) != int:
                end_val = re.sub(bad_pattern, '', str(end_val))
                temp_key = re.findall(num_pattern, str(end_val))[0]
                end_val = end_val.replace(temp_key, str(variable_dict[temp_key]))
                end_val = eval(end_val)

            if item_value['type'] == 'int':
                variable_dict[item_name] = random.randint(start_val, end_val)

            if item_value['type'] == 'float':
                variable_dict[item_name] = random.uniform(start_val, end_val)

        s_variable_dict = {key: str(value) for key, value in variable_dict.items()}
        for key, value in s_variable_dict.items():
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

        tokenized_list_index_string = ' '.join([str(token) for token in tokenized_list_index])


        for key, value in tokenized_dictionary.items():
            if int(value) < 10:
                tokenized_dictionary[key] = "{}{}".format("0", value)
        for key, value in tokenized_dictionary.items():
            tokenized_dictionary[key] = "{}{}".format("_", value)

        equations = data.get('equations')

        for variable_key, variable_value in s_variable_dict.items():
            equations = equations.replace(variable_key, variable_value)

        for tokenized_key, tokenized_value in random_dict.items():
            equations = equations.replace(tokenized_key, tokenized_value)

        for tokenized_key, tokenized_value in tokenized_dictionary.items():
            if tokenized_key not in [',','(',')']:
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
        print(equations)

        tokenized_problem_list_endspaced = tokenized_problem_list
        for i, item in enumerate(tokenized_problem_list):
            tokenized_problem_list_endspaced[i] = "{}{}{}".format(":",item, " ")
        zipped_token_index = ''.join([str(a) + b for a, b in zip(tokenized_list_index, tokenized_problem_list_endspaced)])

        problem_with_equations.append(problem)
        problem_with_equations.append(tokenized_list_index_string)
        problem_with_equations.append(zipped_token_index)
        problem_with_equations.append(equations)

        count += 1

    yaml_dump(args.output_path, problem_with_equations)