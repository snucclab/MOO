import pytest
from json import load, dump
from pathlib import Path
import torch

from common.sys.convert import string_to_text_instance, equation_to_execution
from common.sys.const import EVALUATE_TOKENIZER_PATH, EVALUATE_WEIGHT_DIR
from solver import execution_to_python_code
from model import EPT
from evaluate import Executor


@pytest.fixture(scope='session')
def model():
    model = EPT.create_or_load(str(EVALUATE_WEIGHT_DIR.absolute()))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


@pytest.fixture(scope='session')
def tokenizer():
    return torch.load(str(EVALUATE_TOKENIZER_PATH.absolute()))


@pytest.fixture(scope="session")
def executor():
    executor = Executor(time_limit=0.5)
    return executor


@pytest.fixture(scope='function')
def json_file():
    ALG_PATH = '../resources/ktsde/alg514_kor.json'
    CC_PATH = '../resources/ktsde/CC_kor.json'
    IL_PATH = '../resources/ktsde/IL_kor.json'

    fp = Path(ALG_PATH).open('rt', encoding='utf-8')
    json_file = load(fp)
    return json_file


def test_read_json_file_for_test(model, tokenizer, json_file, executor):
    num_of_items = 0
    num_of_correct = 0

    for item in json_file:
        question = item['sQuestion']
        answer = item['lSolutions'][0]
        instance = string_to_text_instance(question, tokenizer)
        word_info = instance.word_info[0]

        equation = model.forward(text=instance, beam=5)['expression']
        execution = equation_to_execution(equation, batch_index=0, word_size=len(word_info))
        code = execution_to_python_code(execution, word_info, indent=4)
        code, answer_hat = executor.run(code)

        try:
            if answer_hat != "":
                answer_hat = eval(answer_hat)
            else:
                answer_hat = 0
            if type(answer_hat) == list:
                answer_hat = answer_hat[0]
            if abs(answer_hat - answer) < 0.01:
                num_of_correct += 1

            print(f"predicted:{answer_hat}, answer:{answer}")
        finally:
            num_of_items += 1
            continue

    print("=============================================")
    print(f"num of items: {num_of_items}")
    print(f"num of correct: {num_of_correct}")
    print(f"correct rate = {((num_of_correct / num_of_items) * 100)}%")
