import torch

from .const import PAD_ID


class SmoothedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = PAD_ID, reduction: str = 'mean'):
        super().__init__()
        assert reduction in {'mean', 'sum', 'none'}

        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logp: torch.Tensor, target: torch.Tensor, smoothing: float,
                focal: float = 0.0) -> torch.Tensor:
        assert target.shape == logp.shape[:target.dim()] and target.dim() + 1 == logp.dim(), \
            'Shape is different! Target has %s but logp has %s.' % (target.shape, logp.shape)
        assert 0 < smoothing < 1, "Smoothing factor should be in (0.0, 1.0)"

        # Flatten logp and target
        logp = logp.flatten(end_dim=-2)
        is_inf_or_nan = torch.isfinite(logp).logical_not()
        target = target.flatten()

        # Prepare smoothed target
        # Set all probability of the targets which should be ignored as zero.
        # Since D_KL(p, q) = p (log(p) - log(q)), by setting p(x) ≡ 0, these target cannot affect loss anymore.
        smoothed_target = torch.zeros(logp.shape, requires_grad=False, device=target.device)

        # Set target values zero if predicted values are masked with -inf.
        for r, row in enumerate(logp):
            tgt = target[r].item()
            if tgt == self.ignore_index:
                continue

            finites = torch.isfinite(row).to(smoothed_target.device)
            n_cls = finites.sum().item()
            assert n_cls > 0

            smoothing_prob = smoothing / n_cls
            smoothed_target[r].masked_fill_(finites, smoothing_prob)
            smoothed_target[r, tgt] = 1.0 - smoothing

        # Compute loss: - p log q
        smoothed_target = smoothed_target.to(logp.device)
        loss = - smoothed_target * logp.masked_fill(is_inf_or_nan, 0.0)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            not_ignored = target.ne(self.ignore_index).sum(dim=-1)

            # Filter out rows whose elements are all ignored
            loss_sum = loss.sum(dim=-1).masked_select(not_ignored.ne(0))
            not_ignored = not_ignored.masked_select(not_ignored.ne(0))

            return (loss_sum / not_ignored).sum()
