import random
import shutil
from pathlib import Path
from typing import Dict

import numpy
import torch


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
