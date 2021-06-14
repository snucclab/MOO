from pathlib import Path
import yaml
import re
import random
import pyjosa

def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data

def yaml_dump(filepath, data):
    with open(filepath, "w") as file_descriptor:
            yaml.dump(data, file_descriptor)

if __name__ == "__main__":
    filepath = "./3-1-1.yaml"
    data = yaml_loader(filepath)
    print(data)
    problem = data.get('problem')
    print(problem)

    vocabpath = "./vocab.yaml"
    vocab = yaml_loader(vocabpath)
    keys = []
    for key in vocab:
        if key in problem:
            # print(key)
            keys.append(key)
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

    


    problem = problem.replace("<","")
    problem = problem.replace(">","")
    problem = problem.replace(".0","")
    print(problem)



    # problem = pyjosa.replace_josa(problem)

    numbers = []
    for item_name, item_value in data['variable-sampling'].items():
        print(item_name)
        print(item_value)
        for sub_item_name, sub_item_value in data[item_value].items():
            print(sub_item_name, sub_item_value)
                # numbers.append(item_name)

    # print(numbers)
    # for item in numbers: