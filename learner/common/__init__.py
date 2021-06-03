import math
import pickle
from typing import Optional

from ray.tune.resources import Resources
from ray.tune.result import TIMESTEPS_THIS_ITER
from ray.tune.trainable import Trainable
from yaml import dump as yaml_dump

from common.const.model import MDL_ENCODER
from common.const.pad import FLOAT_NAN
from common.dataset import *
from common.tester import Tester
from model import model_loader, EPTPackage, MODEL_CLS
from .const import *
from .util import *


def _read_system_spec() -> str:
    from subprocess import check_output

    return check_output(['bash', str(Path(Path(__file__).parent.parent.parent, 'system_spec.sh'))]).decode('UTF-8')


class TrainerBase(Trainable):
    def __init__(self, config=None, logger_creator=None):
        # Training config
        self._batch_size: int = 0
        # Dataset
        self._dataset: Optional[Dataset] = None
        # Model for learning
        self._model: Optional[EPTPackage] = None
        # Tester
        self._tester: Tester = Tester()
        self._test_rng: Generator = Generator(PCG64(1))
        # Training/Evaluation configuration
        self._train_config: dict = {}
        self._eval_configs: dict = {}
        # Pre-training configuration
        self._pretrain: list = []
        self._pretrain_iter: int = 0
        self._no_pretrain_signal: bool = False

        # Initialize Trainable
        super().__init__(config, logger_creator)

        # Store setup of the current experiment.
        with Path(self.logdir, 'trainer.log').open('w+t', encoding='UTF-8') as fp:
            fp.write('Initializing %s has been finished.\n' % self.__class__.__name__)
            fp.write('\n--------------------  System specification ---------------------\n')
            fp.write(_read_system_spec())
            fp.write('\n-------------------- Trainer configuration ---------------------\n')
            fp.write(yaml_dump(config))
            fp.write('\n--------------------   Model structure     ---------------------\n')
            fp.write(str(self._model))
            fp.write('\n--------------------   Model parameters    ---------------------\n')
            params = [(n, p.numel()) for n, p in self._model.module.named_parameters()]
            fp.write('\n'.join([f'{n}: {p}' for n, p in params]))
            fp.write('\nTOTAL: %s\n' % sum([x for _, x in params]))
            fp.write('\n-------------------- Dataset statistics    ---------------------\n')
            fp.write(yaml_dump(self._dataset.statistics))

    @classmethod
    def default_resource_request(cls, config: dict) -> Resources:
        cls._validate_config(config)

        resource = config[KEY_RESOURCE]
        return Resources(
            cpu=resource[KEY_CPU],
            gpu=resource[KEY_GPU]
        )

    @classmethod
    def _validate_config(cls, config):
        assert KEY_DATASET in config
        assert KEY_MODEL in config
        assert KEY_OPTIMIZER in config
        assert KEY_SEED in config
        assert KEY_BATCH_SZ in config
        assert KEY_RESOURCE in config
        assert KEY_EXPERIMENT in config

        assert type(config[KEY_DATASET]) is str
        assert type(config[KEY_MODEL]) is dict
        assert type(config[KEY_OPTIMIZER]) is dict
        assert type(config[KEY_SEED]) is int
        assert type(config[KEY_BATCH_SZ]) is int
        assert type(config[KEY_RESOURCE]) is dict
        assert type(config[KEY_EXPERIMENT]) is dict
        assert KEY_PRETRAIN_FOR not in config or type(config[KEY_PRETRAIN_FOR]) is list
        assert KEY_PRETRAIN_ITER not in config or type(config[KEY_PRETRAIN_ITER]) is int

        assert config[KEY_BATCH_SZ] > 0

        assert KEY_CPU in config[KEY_RESOURCE]
        assert KEY_GPU in config[KEY_RESOURCE]

        if KEY_GRAD_CLIP in config:
            assert isinstance(config[KEY_GRAD_CLIP], (int, float))

        if KEY_SCHEDULER in config:
            assert type(config[KEY_SCHEDULER]) is dict

    @classmethod
    def get_trial_name(cls, config, trial_id):
        # Naming convention: [Trainer-specific]-[MODEL]-[ID]
        # Get model's trial name
        return '%s-%s' % (config[KEY_MODEL][MODEL_CLS], trial_id)

    def stop(self):
        super().stop()
        self._tester.close()

    def setup(self, config):
        super().setup(config)

        # Setup logging level of transformers to ERROR
        from transformers import logging
        logging.set_verbosity_error()
        self.reset_config(config)

    def reset_config(self, new_config):
        # Set seed
        set_seed(new_config[KEY_SEED])

        # Set batch size
        if self._batch_size == 0 or self._batch_size != new_config[KEY_BATCH_SZ]:
            self._batch_size = new_config[KEY_BATCH_SZ]

        # Read dataset
        if self._dataset is None:
            self._dataset = Dataset(path=new_config[KEY_DATASET], langmodel=new_config[KEY_MODEL][MDL_ENCODER],
                                    seed=new_config[KEY_SEED], formula_field=new_config[KEY_TARGET_FIELD],
                                    number_window=new_config[KEY_WINDOW])
        else:
            self._dataset.reset_seed(new_config[KEY_SEED])

        # Store experiment setup
        if not self._train_config:
            experiments = new_config[KEY_EXPERIMENT]
            self._train_config = experiments.pop(KEY_TRAIN)
            self._eval_configs = experiments

        # Store pretraining setup
        self._no_pretrain_signal = False
        if KEY_PRETRAIN_ITER in new_config and KEY_PRETRAIN_FOR in new_config:
            self._pretrain = new_config[KEY_PRETRAIN_FOR]
            self._pretrain_iter = new_config[KEY_PRETRAIN_ITER]
        else:
            self._pretrain = []
            self._pretrain_iter = 0

        # Load training set
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])

        # Build models
        self._model = model_loader(new_config[KEY_MODEL].copy())
        self._tester.load_tokenizer(new_config[KEY_MODEL][MDL_ENCODER])

        # Build or Re-build optimizer
        step_per_epoch = math.ceil(self._dataset.num_items / self._batch_size)
        self._model.set_max_step(step_per_epoch * 500)  # TODO read max epoch configuration
        self._model.set_optimizer(new_config[KEY_OPTIMIZER])
        self._model.set_grad_clip(new_config.get(KEY_GRAD_CLIP))  # This can be None.
        if KEY_SCHEDULER in new_config:
            new_config[KEY_SCHEDULER]['step_per_epoch'] = step_per_epoch
            self._model.set_scheduler(new_config[KEY_SCHEDULER])

        return True

    def step(self):
        # Prepare metrics
        report = dict()

        # Run scripts before updating
        report['before'] = self._before_update()

        # Run training
        if self.training_iteration < self._pretrain_iter:
            if not self._no_pretrain_signal:
                # If this module does not require pretraining, ignore the pretraining phase
                report['pretrain'] = self._update_module(pretrain=True)
                report[TIMESTEPS_THIS_ITER] = report['pretrain'][TIMESTEPS_THIS_ITER]
            else:
                report[TIMESTEPS_THIS_ITER] = 1
        else:
            report['train'] = self._update_module(pretrain=False)
            report[TIMESTEPS_THIS_ITER] = report['train'][TIMESTEPS_THIS_ITER]

            # Run evaluation periodically
            executed_split = {}
            iter_after_pretrain = (self._iteration + 1) - self._pretrain_iter
            for key, config in self._eval_configs.items():
                period = config[KEY_EVAL_PERIOD]
                split = config.get(KEY_SPLIT_FILE, '')
                if iter_after_pretrain % period == 0 and split:
                    if split in executed_split:
                        # Avoid multiple running on the same split.
                        report[key] = report[executed_split[split]]
                    else:
                        report[key] = self._evaluate(key, config)
                        executed_split[split] = key

        # Run scripts after updating
        report['after'] = self._after_update()

        # Add metric shortcut of development set
        report['dev_correct'] = report.get(KEY_DEV, {}).get('correct', FLOAT_NAN)

        return report

    def __getstate__(self) -> dict:
        # We don't store test_rng since it will be initialized on every evaluation.
        return {
            KEY_MODEL: self._model.state,
            'rng': {
                'numpy': numpy.random.get_state(),
                'random': random.getstate(),
                'torch': {
                    'cpu': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                },
                'dataset': self._dataset.get_rng_state()
            },
            'iteration': self._iteration
        }

    def __setstate__(self, state: dict):
        # Load rng
        random_states = state['rng']
        numpy.random.set_state(random_states['numpy'])
        random.setstate(random_states['random'])
        self._dataset.set_rng_state(random_states['dataset'])

        torch.set_rng_state(random_states['torch']['cpu'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_states['torch']['cuda'])

        # Load iteration
        self._iteration = state['iteration']

        # Load policy
        self._model.load(state[KEY_MODEL])

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir, 'chkpt')
        with checkpoint_path.open('wb') as fp:
            pickle.dump(self.__getstate__(), fp)

        rotate_checkpoint(tmp_checkpoint_dir, max_item=1)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint):
        with Path(checkpoint).open('wb') as fp:
            extra_data = pickle.load(fp)

        self.__setstate__(extra_data)

    def _after_update(self) -> dict:
        return self._model.learning_metric()

    def _record_evaluation_output(self, experiment_name: str, output: dict):
        with Path(self.logdir, '%s.yaml' % experiment_name).open('w+t', encoding='UTF-8') as fp:
            fp.write('# Output of experiment %s in iteration %s.\n' % (experiment_name, self.iteration))
            fp.write('# Total %d items are tested.\n' % len(output['dump']))
            yaml_dump(output, fp)

    def _reset_test_random_generator(self):
        self._test_rng = Generator(PCG64(1))

    def _train(self):
        raise ValueError('Trainer._train() should not be called!')

    def _save(self, tmp_checkpoint_dir):
        raise ValueError('Trainer._save() should not be called!')

    def _restore(self, checkpoint):
        raise ValueError('Trainer._restore() should not be called!')

    def _evaluate(self, name: str, configuration: dict) -> dict:
        raise NotImplementedError()

    def _before_update(self) -> dict:
        raise NotImplementedError()

    def _update_module(self, pretrain: bool) -> dict:
        raise NotImplementedError()
