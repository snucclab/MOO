from argparse import ArgumentParser

from tqdm import tqdm

from common.dataset import Dataset
from common.seed.solve import Solver


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(curriculum=True, imitation=True, ignore_incorrect=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = read_arguments()

    solver = Solver()

    # Load dataset
    for field in {'equations', 'sourceFormEq'}:
        dataset = Dataset(args.dataset, formula_field=field)
        for item in tqdm(dataset._whole_items):
            answers, err = solver.solve(item.expression.to_sympy(item.info.variables), item.info.numbers)
            if err:
                print('Exception (%s) occurred in %s' % (str(err), item.info.item_id))
                continue

            if not solver.check_answer(item.info.answers, answers):
                print('Answer is not same in %s\n\tExpected %s\n\tResulted %s' % (item.info.item_id,
                                                                                  str(item.info.answers), answers))

        print('Finished for %s-%s' % (dataset.get_dataset_name, field))

    solver.close()
