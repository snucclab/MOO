from pathlib import Path
from json import load as json_load, dump as json_save

import torch
from transformers import ElectraTokenizer

from common.sys.const import EVALUATE_INPUT_PATH, EVALUATE_OUTPUT_PATH, EVALUATE_WEIGHT_PATH, EVALUATE_TOKENIZER_PATH
from common.sys.convert import string_to_text_instance, equation_to_execution
from common.sys.key import ANSWER, EQUATION, QUESTION
from evaluate import Executor
from model import EPT
from solver import execution_to_python_code

if __name__ == '__main__':
    # Load model from './weights/checkpoint' using model.EPT.create_or_load()
    model = EPT.create_or_load(EVALUATE_WEIGHT_PATH)
    # Restore tokenizer from pickle
    tokenizer: ElectraTokenizer = torch.load(EVALUATE_TOKENIZER_PATH)
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda:0')
    # Set model as evaluation mode
    model.eval()

    # Read '/home/agc2021/dataset/problemsheet.json' and store (key, text) pairs into problems
    with Path(EVALUATE_INPUT_PATH).open('r+t', encoding='UTF-8') as fp:
        problems = json_load(fp)
    # Initialize code executor
    executor = Executor(time_limit=0.5)

    # Initialize answers as dict
    answers = {}
    # For each text in problems
    for key, text in problems.items():
        # Transform text into common.model.types.Text instance
        instance = string_to_text_instance(text[QUESTION], tokenizer)
        word_info = instance.word_info[0]
        # Generate equation using model.forward()
        equation = model.forward(text=instance)
        # Transform equation into a list of common.solver.types.Execution
        execution = equation_to_execution(equation, batch_index=0, word_size=len(word_info))
        # /* The following two lines will be shared with train_model.py, check_dataset.py */
        # Transform equation into python code using solver.execution_to_python_code()
        code = execution_to_python_code(execution, word_info, indent=4)
        # Execute python code with timeout (0.5s) and get an answer (type: string)
        code, answer = executor.run(code)
        # Set answers[key] as {'answer': answer, 'equation': code}
        answers[key] = {ANSWER: answer, EQUATION: code}

    # Dump answers into './answersheet.json'
    with Path(EVALUATE_OUTPUT_PATH).open('w+t', encoding='UTF-8') as fp:
        json_save(answers, fp)

    # Finalize everything
    executor.close()
