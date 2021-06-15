import yaml
import json
import os
from common.sys.key import QUESTION, ANSWER, EQUATION, EXECUTION

def main():
    yaml_ls = []
    dir_ = '/home/rsk/MOO/yaml_sample'
    for filename in os.listdir(dir_):
        if filename.endswith('.yaml'):
            with open(os.path.join(dir_, filename)) as f:
                content = f.read()
                yaml_el = yaml.load(content) #Set[List[str]]
            yaml_ls += yaml_el

    total_len = len(yaml_ls) // 2
    json_dict = dict()
    for idx in range(total_len):
        json_dict[idx] = {QUESTION : yaml_ls[2*idx], EQUATION: yaml_ls[2*idx+1]}
    
    with open(dir_+'/dataset.json','w',encoding='utf-8') as json_reader:
        json.dump(json_dict,json_reader,ensure_ascii=False)

if __name__ == '__main__':
    main()