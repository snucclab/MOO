from argparse import ArgumentParser
from os import environ

from common.model.const import DEF_ENCODER
from common.sys.convert import equation_to_execution
from common.sys.dataset import Dataset
from common.sys.pattern import *
from evaluate import Executor
from solver import execution_to_python_code


def read_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-dataset', '-i', type=str, required=True,
                        help='Path of dataset.json file')
    parser.add_argument('--seed', '-seed', '-s', type=int, default=9172,
                        help='Random seed for generating items')
    parser.add_argument('--time-limit', '-limit', '-l', type=float, default=0.5,
                        help='Time limit for evaluating python code')
    parser.add_argument('--tokenizer', '-tokenizer', '-z', type=str, default=DEF_ENCODER,
                        help='Pre-trained Tokenizer')
    return parser.parse_args()


if __name__ == '__main__':
    # Read command-line arguments, including datasetpath
    args = read_arguments()
    environ['DEBUG'] = 'True'

    # Read dataset from datasetpath (i.e., generate all required data types for each problem)
    dataset = Dataset(path=args.dataset, langmodel=args.tokenizer, seed=args.seed)
    # Create an executor
    executor = Executor(time_limit=args.time_limit)

    # For item in dataset:
    for single_batch in dataset.get_minibatches(1):
        # Extract equation of the item
        equation = single_batch.expression
        word_info = single_batch.text.word_info[0]
        # Transform equation into a list of common.solver.types.Execution
        execution = equation_to_execution(equation, batch_index=0, word_size=len(word_info))
        # /* The following two lines will be shared with train_model.py, main.py */
        # Transform equation into python code using solver.execution_to_python_code()
        code = execution_to_python_code(execution, word_info, indent=4)
        assert ALL_KOREAN_PATTERN.match(code) is None, \
            '코드에는 한글이 포함될 수 없습니다.\n\t실행한 코드\n%s' % code

        # Execute python code with timeout (0.5s) and get an answer (type: string)
        answer = executor.run(code)
        assert '\n' not in answer, \
            '답은 오직 하나여야 합니다. 지금은 %s개의 답이 출력되었습니다: %s' % (answer.count('\n')+1, answer.split('\n'))
        if NUMBER_PATTERN.fullmatch(answer):
            assert '.' not in answer or UNDER_TWO_DIGIT.fullmatch(answer) is not None, \
                '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(정수이거나 소숫점 이하 두자리)에 맞지 않습니다.' % answer
        elif FRACTION_PATTERN.fullmatch(answer) is None:
            assert ALL_KOREAN_PATTERN.fullmatch(answer) is not None, \
                '출력된 답 "%s"(이)가 대회에서 지정한 출력 형식(텍스트인 경우 기타 기호 없이 한글만)에 맞지 않습니다.' % answer

        # Verify whether the answer is the same as expected one
        # if not same, Report the exception
        assert single_batch.answer[0] == answer, \
            '기대한 답 "%s"(이)가 계산된 답 "%s"(와)과 일치하지 않습니다!\n\t문제번호: %s\n\t실행한 코드\n%s' % \
            (single_batch.answer[0], answer, single_batch.item_id[0], code)

    # Finalize the executor
    executor.close()
