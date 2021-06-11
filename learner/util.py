import random
import shutil
from pathlib import Path
from typing import Dict

import numpy
import torch

from common.model.const import PAD_ID


def rotate_checkpoint(path, max_item: int = 10):
    checkpoints = sorted(Path(path).parent.glob('checkpoint_*'), key=lambda p: int(p.name[11:]))

    # Check if we should delete older checkpoint(s)
    if len(checkpoints) <= max_item:
        return

    for chkpt in checkpoints[:-max_item]:
        # Remove old checkpoints
        shutil.rmtree(chkpt)


def merge_reports(reports) -> Dict[str, float]:
    result = {}
    for key in reports[0]:
        values = [r[key] for r in reports]
        if values and isinstance(values[0], list):
            # Flatten list
            values = sum(values, [])

        if values and isinstance(values[0], dict):
            result[key] = merge_reports(values)
        else:
            result[key] = float(numpy.mean(values))

    return result


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_system_spec() -> str:
    from subprocess import check_output

    return check_output(['bash', str(Path(Path(__file__).parent.parent.parent, 'system_spec.sh'))]).decode('UTF-8')


def num_corrects(target: torch.Tensor, output: torch.Tensor) -> dict:
    assert output.shape[:-1] == target.shape, f'{output.shape} != {target.shape}'

    target = target[:, 1:].cpu()
    predict = output[:, :-1].argmax(dim=-1).cpu()

    # Shape [B, T]
    nonpad = target.ne(PAD_ID)

    correct = predict.eq(target)
    token = correct.logical_and(nonpad).float()
    seq = correct.logical_or(nonpad.logical_not())

    return {
        'token': {
            'corrects': float(token.sum()),
            'total': float(nonpad.sum())
        },
        'seq': {
            'corrects': float(seq.prod(dim=1).sum()),
            'total': int(seq.shape[0]),
            'raw': seq
        }
    }


def accuracy_of(target: torch.Tensor, output: torch.Tensor) -> dict:
    with torch.no_grad():
        counts = num_corrects(target, output)

    token = counts['token']
    seq = counts['seq']

    return {
        'token_acc': token['corrects'] / token['total'] if token['total'] else float('NaN'),
        'seq_acc': seq['corrects'] / seq['total'] if seq['total'] else float('NaN'),
        'raw': seq['raw']
    }
