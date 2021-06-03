import logging
from argparse import ArgumentParser
from collections import defaultdict
from os import cpu_count
from sys import argv

from scipy.stats import sem
from ray import tune, init, shutdown
from ray.tune.trial import Trial
from ray.tune.utils.util import is_nan_or_inf
from torch.cuda import device_count

from common.const.model import *
from common.trial import trial_dirname_creator_generator
from learner.supervised import *
from model import MODELS, MODEL_CLS

CPU_FRACTION = 1.0
GPU_FRACTION = 0.5
ALGORITHM = {
    'supervised': SupervisedTrainer
}


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(simple=False)
    env.add_argument('--name', '-name', type=str, required=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    env.add_argument('--seed', '-seed', type=int, default=1)
    env.add_argument('--experiment-dir', '-exp', type=str, required=True)
    env.add_argument('--beam-desc', '-beamD', type=int, default=5)
    env.add_argument('--beam-expr', '-beamE', type=int, default=3)
    env.add_argument('--window-size', '-win', type=int, default=5)
    env.add_argument('--use-simple', '-simple', action='store_true', dest='simple')

    env.add_argument('--max-iter', '-iter', type=int, default=100)
    env.add_argument('--pretrain-iter', '-iterP', type=int, default=0)
    env.add_argument('--stop-conditions', '-stop', type=str, nargs='*', default=[])

    model = parser.add_argument_group('Model')
    model.add_argument('--model', '-model', type=str, choices=MODELS.keys(), default=['festa'], nargs='+')
    model.add_argument('--encoder', '-enc', type=str, default=DEF_ENCODER)
    model.add_argument('--decoder-hidden', '-decH', type=int, default=0)
    model.add_argument('--decoder-intermediate', '-decI', type=int, default=0)
    model.add_argument('--decoder-layer', '-decL', type=int, default=[6], nargs='+')
    model.add_argument('--decoder-head', '-decA', type=int, default=0)

    log = parser.add_argument_group('Logger setup')
    log.add_argument('--log-path', '-log', type=str, default='./runs')

    work = parser.add_argument_group('Worker setup')
    work.add_argument('--num-cpu', '-cpu', type=float, default=CPU_FRACTION)
    work.add_argument('--num-gpu', '-gpu', type=float, default=GPU_FRACTION)

    setup = parser.add_argument_group('Optimization setup')
    setup.add_argument('--opt-lr', '-lr', type=float, default=[0.00176], nargs='+')
    setup.add_argument('--opt-beta1', '-beta1', type=float, default=0.9)
    setup.add_argument('--opt-beta2', '-beta2', type=float, default=0.999)
    setup.add_argument('--opt-eps', '-eps', type=float, default=1E-8)
    setup.add_argument('--opt-grad-clip', '-clip', type=float, default=10.0)
    setup.add_argument('--opt-warmup', '-warmup', type=float, default=[2], nargs='+')
    setup.add_argument('--batch-size', '-bsz', type=int, default=4)

    alg = parser.add_argument_group('Algorithm setups')
    alg.add_argument('--algorithm', '-algo', type=str, default='supervised', choices=list(ALGORITHM.keys()))

    return parser.parse_args()


def build_experiment_config(args, exp_dir: str = None):
    exp_path = Path(args.experiment_dir if exp_dir is None else exp_dir)
    experiments = {}
    iter_after_pretrain = args.max_iter - args.pretrain_iter
    for file in exp_path.glob('*'):
        if not file.is_file():
            continue

        experiment_dict = {KEY_SPLIT_FILE: str(file.absolute())}
        if file.name != KEY_TRAIN:
            experiment_dict[KEY_BEAM] = args.beam_expr
            experiment_dict[KEY_BEAM_DESC] = args.beam_desc
            experiment_dict[KEY_EVAL_PERIOD] = iter_after_pretrain // 5 if file.name == KEY_DEV else iter_after_pretrain

        experiments[file.name] = experiment_dict
    if KEY_DEV not in experiments:
        experiments[KEY_DEV] = experiments[KEY_TEST].copy()
        experiments[KEY_DEV][KEY_EVAL_PERIOD] = iter_after_pretrain // 5
    return experiments


def build_configuration(args):
    return {
        KEY_SEED: args.seed,
        KEY_BATCH_SZ: args.batch_size,
        KEY_DATASET: str(Path(args.dataset).absolute()),
        KEY_MODEL: {
            MODEL_CLS: tune.grid_search(args.model),
            MDL_ENCODER: args.encoder,
            MDL_DECODER: {
                MDL_D_HIDDEN: args.decoder_hidden,
                MDL_D_INTER: args.decoder_intermediate,
                MDL_D_LAYER: tune.grid_search(args.decoder_layer),
                MDL_D_HEAD: args.decoder_head
            },
            MDL_DESCRIBER: {
                MDL_ENCODER: args.encoder
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
            'lr': tune.grid_search(args.opt_lr),
            'betas': (args.opt_beta1, args.opt_beta2),
            'eps': args.opt_eps,
            'debias': True
        },
        KEY_SCHEDULER: {
            'type': 'warmup-linear',
            'num_warmup_epochs': tune.grid_search(args.opt_warmup),
            'num_total_epochs': args.max_iter
        },
        KEY_PRETRAIN_ITER: args.pretrain_iter,
        KEY_PRETRAIN_FOR: ['num_desc', 'var_desc', 'var_desc_flat'],
        KEY_WINDOW: args.window_size,
        KEY_TARGET_FIELD: 'simpleEq' if args.simple else 'equations'
    }


def build_stop_condition(args):
    cond_dict = dict(training_iteration=args.max_iter)
    for condition in args.stop_conditions:
        key, value = condition.split('=')
        cond_dict[key] = float(value)

    return cond_dict


def get_experiment_name(args):
    from datetime import datetime
    now = datetime.now().strftime('%m%d%H%M%S')
    field = 'simpleEq' if args.simple else 'equations'
    return f'{args.algorithm}_{Path(args.dataset).stem}_{field}_{args.name}_{now}'


if __name__ == '__main__':
    args = read_arguments()
    if not Path(args.log_path).exists():
        Path(args.log_path).mkdir(parents=True)

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path(args.log_path, 'meta.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger('Hyperparameter Optimization')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('========================= CMD ARGUMENT =============================')
    logger.info(' '.join(argv))
    init(num_cpus=cpu_count(), num_gpus=device_count())

    algorithm_cls = ALGORITHM[args.algorithm]
    experiment_name = get_experiment_name(args)

    stop_condition = build_stop_condition(args)
    analysis = tune.run(algorithm_cls, name=experiment_name, stop=stop_condition,
                        config=build_configuration(args), local_dir=args.log_path, checkpoint_at_end=True,
                        checkpoint_freq=args.max_iter // 5, reuse_actors=True,
                        trial_dirname_creator=trial_dirname_creator_generator(), raise_on_failed_trial=False,
                        metric='dev_correct', mode='max')

    # Record trial information
    logger.info('========================= DEV. RESULTS =============================')
    logger.info('Hyperparameter search is finished!')
    trials: List[Trial] = analysis.trials
    best_scores = defaultdict(float)
    best_configs = {}
    best_trials = {}

    for trial in trials:
        if trial.status != Trial.TERMINATED:
            logger.info('\tTrial %10s (%-40s): FAILED', trial.trial_id, trial.experiment_tag)
            continue

        last_score = trial.last_result['dev_correct']
        logger.info('\tTrial %10s (%-40s): Correct %.4f on dev. set', trial.trial_id, trial.experiment_tag, last_score)

        if is_nan_or_inf(last_score):
            continue

        model_cls = trial.config[KEY_MODEL][MODEL_CLS]
        if best_scores[model_cls] < last_score:
            best_scores[model_cls] = last_score
            best_configs[model_cls] = trial.config
            best_trials[model_cls] = trial

    # Record the best configuration
    logpath = Path(analysis.best_logdir).parent
    for cls, config in best_configs.items():
        logger.info('--------------------------------------------------------')
        logger.info('Found best configuration for %s (scored %.4f)', cls, best_scores[cls])
        logger.info(repr(config))
        logger.info('--------------------------------------------------------')
        with Path(logpath, 'best_config_%s.pkl' % cls).open('wb') as fp:
            pickle.dump(config, fp)
        with Path(logpath, 'best_config_%s.yaml' % cls).open('w+t') as fp:
            yaml_dump(config, fp)

    # If experiment_dir has sibling directories, apply the specified configuration for all the other directories.
    exp_dir = Path(args.experiment_dir)
    exp_siblings = []
    for p in exp_dir.parent.iterdir():
        if p.samefile(exp_dir):
            continue

        dir_creator = trial_dirname_creator_generator(suffix=p.stem)
        for cls, config in best_configs.items():
            config = config.copy()
            config.update({KEY_EXPERIMENT: build_experiment_config(args, str(p))})

            exp_siblings.append(tune.Experiment(name=experiment_name, run=algorithm_cls, stop=stop_condition,
                                                config=config, local_dir=args.log_path,
                                                trial_dirname_creator=dir_creator, checkpoint_at_end=True,
                                                checkpoint_freq=args.max_iter // 5))

    # Collect the result of experiments that used the best config
    all_exps = {key: [trial] for key, trial in best_trials.items()}
    if len(exp_siblings) > 1:
        # Run other sibling experiments and store it
        analysis = tune.run(exp_siblings, reuse_actors=True, raise_on_failed_trial=False, metric='dev_correct',
                            mode='max')

        for trial in analysis.trials:
            if trial.status != Trial.TERMINATED:
                continue

            model_cls = trial.config[KEY_MODEL][MODEL_CLS]
            all_exps[model_cls].append(trial)

    # Compute average and standard error.
    logger.info('========================= TEST RESULTS =============================')
    for cls, trials in all_exps.items():
        logger.info('For model %10s', cls)
        logger.info('Order of trials:')
        trials = sorted(trials, key=lambda t: t.config[KEY_EXPERIMENT][KEY_TRAIN][KEY_SPLIT_FILE])
        metrics = defaultdict(list)

        # Collect metrics
        for trial in trials:
            logger.info('\t%s', trial.config[KEY_EXPERIMENT][KEY_TRAIN][KEY_SPLIT_FILE])

            test_metrics = trial.last_result['test']
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[metric].append(value)

        for metric, values in metrics.items():
            logger.info('Metric %10s: %.4f ± %.4f', metric, mean(values), sem(values))
            logger.info('\tscores: [%s]', ','.join(['%.4f' % v for v in values]))
        logger.info('--------------------------------------------------------------------')

    shutdown()
