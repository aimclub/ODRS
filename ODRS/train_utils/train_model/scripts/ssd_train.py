import os
from pathlib import Path


def train_ssd(CONFIG_PATH):
    """
    Runs SSD training using the parameters specified in the config.
    """
    file = Path(__file__).resolve()
    os.system(
        f'python3 {file.parents[1]}/models/PyTorch-SSD/train.py'
        f" --cfg {CONFIG_PATH}"
        f" --logdir {os.path.dirname(CONFIG_PATH)}/exp")