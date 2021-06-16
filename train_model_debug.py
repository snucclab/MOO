import logging
from argparse import ArgumentParser
from sys import argv

from common.model.const import *
from learner import *

CPU_FRACTION = 1.0
GPU_FRACTION = 0.5


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(simple=False)
    env.add_argument('--name', '-name', type=str, required=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    env.add_argument('--seed', '-seed', type=int, default=1)
    env.add_argument('--beam', '-beam', type=int, default=3)

    env.add_argument('--max-iter', '-iter', type=int, default=100)

    model = parser.add_argument_group('Model')
    model.add_argument('--encoder', '-enc', type=str, default=DEF_ENCODER)
    model.add_argument('--decoder-hidden', '-decH', type=int, default=0)
    model.add_argument('--decoder-intermediate', '-decI', type=int, default=0)
    model.add_argument('--decoder-layer', '-decL', type=int, default=6)
    model.add_argument('--decoder-head', '-decA', type=int, default=0)

    log = parser.add_argument_group('Logger setup')
    log.add_argument('--log-path', '-log', type=str, default='./runs')

    work = parser.add_argument_group('Worker setup')
    work.add_argument('--num-cpu', '-cpu', type=float, default=CPU_FRACTION)
    work.add_argument('--num-gpu', '-gpu', type=float, default=GPU_FRACTION)

    setup = parser.add_argument_group('Optimization setup')
    setup.add_argument('--opt-lr', '-lr', type=float, default=0.00176)
    setup.add_argument('--opt-beta1', '-beta1', type=float, default=0.9)
    setup.add_argument('--opt-beta2', '-beta2', type=float, default=0.999)
    setup.add_argument('--opt-eps', '-eps', type=float, default=1E-8)
    setup.add_argument('--opt-grad-clip', '-clip', type=float, default=10.0)
    setup.add_argument('--opt-warmup', '-warmup', type=float, default=2)
    setup.add_argument('--batch-size', '-bsz', type=int, default=4)

    return parser.parse_args()


def build_experiment_config(args):
    exp_path = Path(args.dataset).parent / 'split'
    experiments = {}
    for file in exp_path.glob('*'):
        if not file.is_file():
            continue

        experiment_dict = {KEY_SPLIT_FILE: str(file.absolute())}
        if file.name != KEY_TRAIN:
            experiment_dict[KEY_BEAM] = args.beam
            experiment_dict[KEY_EVAL_PERIOD] = args.max_iter // 5 if file.name == KEY_DEV else args.max_iter

        experiments[file.name] = experiment_dict
    if KEY_DEV not in experiments:
        experiments[KEY_DEV] = experiments[KEY_TEST].copy()
        experiments[KEY_DEV][KEY_EVAL_PERIOD] = args.max_iter // 5
    return experiments


def build_configuration(args):
    return {
        KEY_SEED: args.seed,
        KEY_BATCH_SZ: args.batch_size,
        KEY_DATASET: str(Path(args.dataset).absolute()),
        KEY_MODEL: {
            MDL_ENCODER: {
                MDL_ENCODER: args.encoder
            },
            MDL_DECODER: {
                MDL_D_HIDDEN: args.decoder_hidden,
                MDL_D_INTER: args.decoder_intermediate,
                MDL_D_LAYER: args.decoder_layer,
                MDL_D_HEAD: args.decoder_head
            }
        },
        KEY_RESOURCE: {
            KEY_GPU: args.num_gpu,
            KEY_CPU: args.num_cpu
        },
        KEY_EXPERIMENT: build_experiment_config(args),
        KEY_GRAD_CLIP: args.opt_grad_clip,
        KEY_OPTIMIZER: {
            'type': 'lamb',
            'lr': args.opt_lr,
            'betas': (args.opt_beta1, args.opt_beta2),
            'eps': args.opt_eps,
            'debias': True
        },
        KEY_SCHEDULER: {
            'type': 'warmup-linear',
            'num_warmup_epochs': args.opt_warmup,
            'num_total_epochs': args.max_iter
        }
    }


def get_experiment_name(args):
    from datetime import datetime
    now = datetime.now().strftime('%m%d%H%M%S')
    return f'{args.name}_{now}'


if __name__ == '__main__':
    args = read_arguments()
    if not Path(args.log_path).exists():
        Path(args.log_path).mkdir(parents=True)

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path(args.log_path, 'meta.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger('Debug Test')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('========================= CMD ARGUMENT =============================')
    logger.info(' '.join(argv))

    experiment_name = get_experiment_name(args)
    trainer = SupervisedTrainer(build_configuration(args))

    for _ in range(args.max_iter):
        trainer.step()

    trainer.save_checkpoint(str(Path(args.log_path, 'test-checkpoint')))
    trainer.cleanup()
