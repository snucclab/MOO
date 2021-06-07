from typing import Type

from ray.tune.trial import Trial


def trial_dirname_creator_generator(suffix: str = ''):
    from learner.common import TrainerBase

    _subclasses = []
    _class_to_visit = [TrainerBase]
    while _class_to_visit:
        _cls = _class_to_visit.pop()
        _subclasses.append((_cls.__name__, _cls))
        _class_to_visit += _cls.__subclasses__()

    _subclasses = sorted(_subclasses, key=lambda t: len(t[0]), reverse=True)

    def get_trainable_class(trainable: str) -> Type[TrainerBase]:
        for cls_name, cls in _subclasses:
            if trainable.startswith(cls_name):
                return cls

        raise ValueError('%s is not a subclass of learner.base.TrainerBase' % trainable)

    def trial_dirname_creator(trial: Trial) -> str:
        trial_id = trial.trial_id
        trainable = trial.trainable_name
        config = trial.config

        cls = get_trainable_class(trainable)
        name = cls.get_trial_name(config, trial_id)

        if suffix:
            name += '_' + suffix

        return name

    return trial_dirname_creator


__all__ = ['trial_dirname_creator_generator']
