from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLinearDecay(LambdaLR):

    def __init__(self, optimizer: Optimizer, num_warmup_epochs: int, num_total_epochs: int, step_per_epoch: int,
                 last_epoch=-1) -> None:
        self.num_warmup_steps = num_warmup_epochs * step_per_epoch
        self.num_total_steps = num_total_epochs * step_per_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))

        return max(
            0.0,
            float(self.num_total_steps - current_step) / float(max(1, self.num_total_steps - self.num_warmup_steps))
        )


class LinearWarmupNoDecay(LambdaLR):
    def __init__(self, optimizer: Optimizer, num_warmup_epochs: int, step_per_epoch: int, last_epoch=-1) -> None:
        self.num_warmup_steps = num_warmup_epochs * step_per_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0
