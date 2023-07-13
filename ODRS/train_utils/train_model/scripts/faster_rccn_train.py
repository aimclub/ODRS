import os
from pathlib import Path


def train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, IMG_SIZE):
    """
    Runs faster-rccn training using the parameters specified in the config.
    """
    file = Path(__file__).resolve()
    os.system(
        f'python3 {file.parents[1]}/models/fastercnn-pytorch-training-pipeline/train.py'
        f" --data {CONFIG_PATH}"
        f" --epochs {EPOCHS}"
        f" --batch {BATCH_SIZE}"
        f" --model fasterrcnn_resnet50_fpn" +
        f" --name {os.path.dirname(CONFIG_PATH)}")
