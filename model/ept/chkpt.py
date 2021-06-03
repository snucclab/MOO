from pathlib import Path

import torch
from torch import nn


class CheckpointingModule(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    @classmethod
    def checkpoint_path(cls, directory: str):
        return Path(directory, '%s.pt' % cls.__name__)

    @classmethod
    def create_or_load(cls, path: str = None, **config):
        state = None

        if path is not None and cls.checkpoint_path(path).exists():
            with cls.checkpoint_path(path).open('rb') as fp:
                load_preset = torch.load(fp)

            config = load_preset['config']
            state = load_preset['state']

        model = cls(**config)
        if state is not None:
            model.load_state_dict(state)

        return model

    def save(self, directory: str):
        with self.checkpoint_path(directory).open('wb') as fp:
            torch.save({
                'config': self.config,
                'state': self.state_dict()
            }, fp)


__all__ = ['CheckpointingModule']
