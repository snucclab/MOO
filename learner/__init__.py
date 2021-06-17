import math
import pickle
from collections import defaultdict
from decimal import Decimal
from typing import Optional

from ray.tune.resources import Resources
from ray.tune.result import TIMESTEPS_THIS_ITER
from ray.tune.trainable import Trainable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from yaml import dump as yaml_dump

from common.model.const import MDL_ENCODER, FLOAT_NAN
from common.model.loss import SmoothedCrossEntropyLoss
from common.sys.convert import equation_to_execution
from common.sys.dataset import *
from common.sys.key import WORD
from common.sys.pattern import NUMBER_BEGIN_PATTERN
from evaluate import Executor
from model import EPT
from solver import execution_to_python_code
from .const import *
from .util import *

SMOOTHED_CROSS_ENTROPY_LOSS = SmoothedCrossEntropyLoss(ignore_index=PAD_ID)


class SupervisedTrainer(Trainable):
    def __init__(self, config=None, logger_creator=None):
        # Training config
        self._batch_size: int = 0
        # Dataset
        self._dataset: Optional[Dataset] = None
        # Model for learning
        self._model: Optional[EPT] = None
        # Tester
        self._tester: Executor = Executor()
        # Training/Evaluation configuration
        self._train_config: dict = {}
        self._eval_configs: dict = {}
        # Optimization config
        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[LambdaLR] = None
        self._grad_clip: float = 0.0

        # Initialize Trainable
        super().__init__(config, logger_creator)

        # Store setup of the current experiment.
        with Path(self.logdir, 'trainer.log').open('w+t', encoding='UTF-8') as fp:
            fp.write('Initializing %s has been finished.\n' % self.__class__.__name__)
            fp.write('\n--------------------  System specification ---------------------\n')
            fp.write(read_system_spec())
            fp.write('\n-------------------- Trainer configuration ---------------------\n')
            fp.write(yaml_dump(config))
            fp.write('\n--------------------   Model structure     ---------------------\n')
            fp.write(str(self._model))
            fp.write('\n--------------------   Model parameters    ---------------------\n')
            params = [(n, p.numel()) for n, p in self._model.named_parameters()]
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

        assert config[KEY_BATCH_SZ] > 0

        assert KEY_CPU in config[KEY_RESOURCE]
        assert KEY_GPU in config[KEY_RESOURCE]

        if KEY_GRAD_CLIP in config:
            assert isinstance(config[KEY_GRAD_CLIP], (int, float))

        if KEY_SCHEDULER in config:
            assert type(config[KEY_SCHEDULER]) is dict

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
        self._batch_size = new_config[KEY_BATCH_SZ]

        # Read dataset
        if self._dataset is None:
            self._dataset = Dataset(path=new_config[KEY_DATASET],
                                    langmodel=new_config[KEY_MODEL][MDL_ENCODER][MDL_ENCODER],
                                    seed=new_config[KEY_SEED])
        else:
            self._dataset.reset_seed(new_config[KEY_SEED])

        # Store experiment setup
        if not self._train_config:
            experiments = new_config[KEY_EXPERIMENT]
            self._train_config = experiments.pop(KEY_TRAIN)
            self._eval_configs = experiments

        # Load training set
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])

        # Build models
        self._model = EPT(**new_config[KEY_MODEL])
        if torch.cuda.is_available():
            self._model.cuda()

        # Build or Re-build optimizer
        step_per_epoch = math.ceil(self._dataset.num_items / self._batch_size)
        self._set_optimizer(new_config[KEY_OPTIMIZER])
        self._set_grad_clip(new_config.get(KEY_GRAD_CLIP))  # This can be None.
        if KEY_SCHEDULER in new_config:
            new_config[KEY_SCHEDULER]['step_per_epoch'] = step_per_epoch
            self._set_scheduler(new_config[KEY_SCHEDULER])

        return True

    def step(self):
        # Prepare metrics
        report = dict()

        # Run scripts before updating
        report['before'] = self._before_update()

        # Run training
        report['train'] = self._update_module()
        report[TIMESTEPS_THIS_ITER] = report['train'][TIMESTEPS_THIS_ITER]

        # Run evaluation periodically
        executed_split = {}
        iter_after_pretrain = self._iteration + 1
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
        return {
            KEY_MODEL: self._model.state_dict(),
            'rng': {
                'numpy': numpy.random.get_state(),
                'random': random.getstate(),
                'torch': {
                    'cpu': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                },
                'dataset': self._dataset.get_rng_state()
            },
            'iteration': self._iteration,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict()
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
        self._model.load_state_dict(state[KEY_MODEL])
        if self._optimizer is not None and 'optimizer' in state:
            self._optimizer.load_state_dict(state['optimizer'])
        if self._scheduler is not None and 'scheduler' in state:
            self._scheduler.load_state_dict(state['scheduler'])

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir, 'chkpt')
        with checkpoint_path.open('wb') as fp:
            pickle.dump(self.__getstate__(), fp)

        # Save model & tokenizer
        self._model.save(tmp_checkpoint_dir)
        with Path(tmp_checkpoint_dir, 'tokenizer.pt').open('wb') as fp:
            torch.save(self._dataset.tokenizer, fp)

        rotate_checkpoint(tmp_checkpoint_dir, max_item=1)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint):
        with Path(checkpoint).open('wb') as fp:
            extra_data = pickle.load(fp)

        self.__setstate__(extra_data)

    def _set_grad_clip(self, param):
        assert param is None or param >= 0
        self._grad_clip = param

    def _set_optimizer(self, kwargs):
        name = kwargs.pop('type')
        if name == 'lamb':
            from torch_optimizer import Lamb
            cls = Lamb
        elif name == 'radam':
            from torch_optimizer import RAdam
            cls = RAdam
        elif name == 'adabound':
            from torch_optimizer import AdaBound
            cls = AdaBound
        elif name == 'yogi':
            from torch_optimizer import Yogi
            cls = Yogi
        elif name == 'adamw':
            from transformers import AdamW
            cls = AdamW
        else:
            from torch.optim.sgd import SGD
            cls = SGD

        self._optimizer = cls([params
                               for key, params in self._model.named_parameters()
                               if 'encoder.model.embeddings' not in key], **kwargs)

    def _set_scheduler(self, kwargs):
        name = kwargs.pop('type')
        if name == 'warmup-linear':
            from .scheduler import LinearWarmupLinearDecay
            self._scheduler = LinearWarmupLinearDecay(self._optimizer, **kwargs)
        elif name == 'warmup-constant':
            from .scheduler import LinearWarmupNoDecay
            self._scheduler = LinearWarmupNoDecay(self._optimizer, **kwargs)
        else:
            self._scheduler = None

    def _after_backprop(self):
        if self._grad_clip is not None and self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)

        if self._optimizer is not None:
            self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

        self._model.zero_grad()

    def _after_update(self) -> dict:
        metric = {}
        if self._scheduler is not None:
            metric['lr'] = max(self._scheduler.get_last_lr())
        elif self._optimizer is not None:
            metric['lr'] = max(group['lr'] for group in self._optimizer.param_groups)

        weight_sizes = defaultdict(list)
        for key, weight in self._model.named_parameters():
            wkey = 'other'
            if 'encoder' in key:
                wkey = 'encoder'
            elif 'decoder' in key:
                wkey = 'decoder'
            elif 'action' in key:
                wkey = 'action'

            if '_embedding' in key:
                wkey += '_embed'
            if 'bias' in key:
                wkey += '_bias'

            weight_sizes[wkey].append(weight.detach().abs().mean().item())

        metric['weight'] = {key: sum(values) / len(values) if values else FLOAT_NAN
                            for key, values in weight_sizes.items()}

        return metric

    def _record_evaluation_output(self, experiment_name: str, output: dict):
        with Path(self.logdir, '%s.yaml' % experiment_name).open('w+t', encoding='UTF-8') as fp:
            fp.write('# Output of experiment %s in iteration %s.\n' % (experiment_name, self.iteration))
            fp.write('# Total %d items are tested.\n' % len(output['dump']))
            yaml_dump(output, fp)

    def _train(self):
        raise ValueError('Trainer._train() should not be called!')

    def _save(self, tmp_checkpoint_dir):
        raise ValueError('Trainer._save() should not be called!')

    def _restore(self, checkpoint):
        raise ValueError('Trainer._restore() should not be called!')

    def _evaluate(self, name: str, configuration: dict) -> dict:
        self._dataset.select_items_with_file(configuration[KEY_SPLIT_FILE])
        self._model.eval()

        results = []
        with torch.no_grad():
            batches = self._dataset.get_minibatches(self._batch_size)

            for batch in batches:
                # ---- Input ----
                # text: Text [B, S]
                # beam: int
                # beam_desc: int
                output = self._model(text=batch.text.to(self._model.device), beam=configuration[KEY_BEAM])
                output = output['expression']
                word_size = max(len(word_info) for word_info in batch.text.word_info)

                for i in range(output.operator.shape[0]):
                    word_info = batch.text.word_info[i]
                    expected = batch.answer[i]
                    # Transform equation into a list of common.solver.types.Execution
                    execution = equation_to_execution(output, batch_index=i, word_size=word_size)
                    # /* The following two lines will be shared with train_model.py, check_dataset.py */
                    # Transform equation into python code using solver.execution_to_python_code()
                    code = execution_to_python_code(execution, word_info, indent=4)
                    # Execute python code with timeout (0.5s) and get an answer (type: string)
                    code, answer = self._tester.run(code)

                    if NUMBER_BEGIN_PATTERN.fullmatch(expected) and NUMBER_BEGIN_PATTERN.fullmatch(answer):
                        correct = abs(Decimal(expected) - Decimal(answer)) < 1E-2
                    else:
                        correct = answer == expected

                    results.append({
                        'correct': correct,
                        'answer': answer,
                        'expected': expected,
                        'execution': [str(x) for x in execution],
                        'code': code,
                        'item_id': batch.item_id[i],
                        'text': ' '.join('_%02d:%s' % (t, tok[WORD]) for t, tok in enumerate(word_info))
                    })

            results = {
                'dump': results,
                'correct': sum([x['correct'] for x in results]) / len(results)
            }
            self._record_evaluation_output(name, results)

        # Remove 'dump' key before returning
        results.pop('dump')
        return results

    def _before_update(self) -> dict:
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])
        self._model.train()
        return {}

    def _update_module(self) -> dict:
        reports = []
        batch_gen = list(self._dataset.get_minibatches(self._batch_size))
        for batch in batch_gen:
            # ---- Input ----
            # text: Text [B, S]
            # expression: Expression [B, T]
            # ---- Output ----
            # expression: ExpressionPrediction [B, T]
            # num_desc?: B-List of Prediction [N, D]
            # var_desc?: B-List of Prediction [V, D] or Prediction [B, VD]
            # var_target?: Label [B, VD]
            out = self._model(text=batch.text.to(self._model.device), 
                              expression=batch.expression.to(self._model.device))
            out = out['prediction']
            tgt = batch.expression.to(out.device)

            # Compute accuracy of tokens
            # Compute loss
            report = {}
            losses = {}

            losses['operator'] = SMOOTHED_CROSS_ENTROPY_LOSS(out.operator[:, :-1], tgt.operator[:, 1:], smoothing=0.01)
            report.update(**{key + '_operator': value
                             for key, value in accuracy_of(tgt.operator, out.operator).items()})
            for j, (o_j, t_j) in enumerate(zip(out.operands, tgt.operands)):
                losses['operand_%s' % j] = SMOOTHED_CROSS_ENTROPY_LOSS(o_j[:, :-1], t_j[:, 1:], smoothing=0.01)
                report.update(**{key + '_operand%s' % j: value
                                 for key, value in accuracy_of(t_j, o_j).items()})

            with torch.no_grad():
                all_raw = [tensor for key, tensor in report.items() if key.startswith('raw_')]
                report = {key: tensor
                          for key, tensor in report.items() if not key.startswith('raw_')}
                report['seq_acc_all'] = float(torch.stack(all_raw).prod(dim=0).prod(dim=1).float().mean())

            # Build sum of losses
            total_loss = sum(losses.values())
            losses['total'] = total_loss
            report.update({'loss_' + key: value for key, value in losses.items()})

            # Add to report (cast to float to avoid memory leak)
            reports.append({key: float(value) for key, value in report.items()})

            # Run Backward prop.
            total_loss.backward()
            self._after_backprop()

        report = merge_reports(reports)
        report[TIMESTEPS_THIS_ITER] = len(reports)
        return report
