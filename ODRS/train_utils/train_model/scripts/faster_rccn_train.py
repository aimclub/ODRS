import os
from pathlib import Path

def train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, IMG_SIZE):
    """
    Runs faster-rccn training using the parameters specified in the config.

    """
    file = Path(__file__).resolve()
    os.system(
        f'python3 {file.parents[1]}/models/fastercnn-pytorch-training-pipeline/train.py'
        ' --data ' +
        CONFIG_PATH +
        ' --epochs ' +
        EPOCHS +
        ' --batch ' +
        BATCH_SIZE +
        ' --model fasterrcnn_resnet50_fpn ' +
        # ' --imgsz ' +
        # IMG_SIZE +
        ' --name ' +
        os.path.dirname(CONFIG_PATH))