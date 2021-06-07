from argparse import ArgumentParser

from tqdm import tqdm

from common.dataset import Dataset
from common.seed.solve import Solver


def read_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-data', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # Read command-line arguments, including datasetpath
    args = read_arguments()

    # Read dataset from datasetpath (i.e., generate all required data types for each problem)


    #
    # For item in dataset:
    # 	Extract execution of the item
    # 	Transform equation into a list of common.solver.types.Execution
    # 	/* The following two lines will be shared with train_model.py, main.py */
    # 	Transform equation into python code using solver.execution_to_python_code()
    # 	Execute python code with timeout (0.5s) and get an answer (type: string)
    # 	Verify whether the answer is the same as expected one
    # 	if not same
    # 		Report the exception
