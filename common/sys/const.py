# Note: Do not import other libraries here, except common.sys.key
from pathlib import Path

MODULE_ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
MODULE_RSC_PATH = MODULE_ROOT_PATH / 'resources'

EVALUATE_INPUT_PATH = '/home/agc2021/dataset/problemsheet.json'
EVALUATE_OUTPUT_PATH = MODULE_ROOT_PATH / 'answersheet.json'
EVALUATE_WEIGHT_DIR = MODULE_ROOT_PATH / 'weight'
EVALUATE_WEIGHT_PATH = EVALUATE_WEIGHT_DIR / 'EPT.pt'
EVALUATE_TOKENIZER_PATH = EVALUATE_WEIGHT_DIR / 'tokenizer.pt'
