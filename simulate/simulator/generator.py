from argparse import ArgumentParser
from pathlib import Path
import yaml
import re
import random
import josa_converter
# from convert import tokenize_string

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

def yaml_dump(filepath, data):
    with open(filepath, "w", encoding="utf-8") as file_descriptor:
        yaml.dump(data, file_descriptor, allow_unicode=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--template_path', '-t', type=str, default='./3-3-2.yaml')
    parser.add_argument('--samples_per_template', '-N', type=int, default=100)
    parser.add_argument('--output_path', '-o', type=str, default='./output/3-3-2_samples.yaml')
    args = parser.parse_args()

    count = 1
    problem_with_equations = []
    while count != args.samples_per_template+1:
        data = yaml_loader(args.template_path)
        # print(data)
        problem = data.get('problem')
        # print(problem)

        vocabpath = "./vocab.yaml"
        vocab = yaml_loader(vocabpath)
        keys = []
        for key in vocab:
            if key in problem:
                # print(key)
                keys.append(key)
            else:
                pass
        print(keys)

        random_dict = {}
        for key in keys:
            random_dict[key]=random.choice(vocab.get(key))
        print(random_dict)

        for key, value in random_dict.items():
            problem = problem.replace(key, value)
        print(problem)
        # result = re.search(r"\[([A-Za-z0-9_]+)\]", problem)
        # result = problem.split('<', 1)[1].split('>')[0]
        # print(result)

        numbers = []
        variable_dict = {}
        for item_name, item_value in data['variable-sampling'].items():
            print(item_name)
            print(item_value)

            if item_value['type'] == 'int':
                print(item_value['type'])
                variable_dict[item_name] = random.randint(item_value['range'][0], item_value['range'][1])

            if item_value['type'] == 'float':
                variable_dict[item_name] = random.uniform(item_value['range'][0], item_value['range'][1])
        print(variable_dict)
        s_variable_dict = {key: str(value) for key, value in variable_dict.items()}
        for key, value in s_variable_dict.items():
            problem = problem.replace(key, value)
        print(problem)
        # print(numbers)
        # for item in numbers:

        problem = problem.replace("<", "")
        problem = problem.replace(">", "")
        problem = problem.replace(".0", "")
        problem = problem.rstrip()
        problem = str(count)+". "+problem

        problem = josa_converter.replace_josa(problem)
        
        print(problem)
        
        problem_with_equations.append(problem)

        # tokenized = tokenize_string(problem)

        equations = data.get('equations')
        print(equations)


        for variable_key, variable_value in s_variable_dict.items():
            equations = equations.replace(variable_key, variable_value)

        equations = equations.replace("<", "")
        equations = equations.replace(">", "")

        print(equations)

        problem_with_equations.append(equations)

        count += 1

    # print(*problems, sep = "\n")
    # print(problems)

    # equations =
    #
    # print(equations)

    yaml_dump(args.output_path, problem_with_equations)