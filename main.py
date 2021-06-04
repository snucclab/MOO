from pathlib import Path
from json import load as json_load

import torch
from transformers import AutoTokenizer, ElectraTokenizer

from common.sys.const import EVALUATE_INPUT_PATH, EVALUATE_OUTPUT_PATH, EVALUATE_WEIGHT_PATH, EVALUATE_TOKENIZER_PATH
from model import EPT

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

    # Initialize answers as dict
    answers = {}
    # For each text in problems
    for key, text in problems.items():
        pass
        # Transform text into common.model.types.Text instance
        # Generate equation using model.forward()
        # Transform equation into a list of common.solver.types.Execution
        # /* The following two lines will be shared with train_model.py, check_dataset.py */
        # Transform equation into python code using solver.execution_to_python_code()
        # Execute python code with timeout (0.5s) and get an answer (type: string)
        # Set answers[key] as {'answer': answer, 'equation': code}

    # Dump answers into './answersheet.json'
    # Finalize everything
