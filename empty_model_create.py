from argparse import ArgumentParser

from ray import tune
from transformers import AutoTokenizer

from common.model.const import *
from common.sys.const import EVALUATE_WEIGHT_DIR
from learner import *

CPU_FRACTION = 1.0
GPU_FRACTION = 0.5


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(simple=False)
    model = parser.add_argument_group('Model')
    model.add_argument('--encoder', '-enc', type=str, default=DEF_ENCODER)
    model.add_argument('--decoder-hidden', '-decH', type=int, default=0)
    model.add_argument('--decoder-intermediate', '-decI', type=int, default=0)
    model.add_argument('--decoder-layer', '-decL', type=int, default=[6], nargs='+')
    model.add_argument('--decoder-head', '-decA', type=int, default=0)

    return parser.parse_args()


def build_model_config(args):
    return {
        MDL_ENCODER: {
            MDL_ENCODER: args.encoder
        },
        MDL_DECODER: {
            MDL_D_HIDDEN: args.decoder_hidden,
            MDL_D_INTER: args.decoder_intermediate,
            MDL_D_LAYER: tune.grid_search(args.decoder_layer),
            MDL_D_HEAD: args.decoder_head
        }
    }


if __name__ == '__main__':
    args = read_arguments()
    if not Path(EVALUATE_WEIGHT_DIR).exists():
        Path(EVALUATE_WEIGHT_DIR).mkdir(parents=True)

    model = EPT(**build_model_config(args))
    model.save(EVALUATE_WEIGHT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    with Path(EVALUATE_WEIGHT_DIR, 'tokenizer.pt').open('wb') as fp:
        torch.save(tokenizer, fp)
