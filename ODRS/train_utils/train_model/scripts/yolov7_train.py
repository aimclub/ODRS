import os
from pathlib import Path

def train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov7 training using the parameters specified in the config.
    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    """
    file = Path(__file__).resolve()
    os.system(f'OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node {GPU_COUNT} {file.parents[1]}/models/yolov7/train.py'
    ' --device ' +
    SELECT_GPU +
    ' --batch-size ' +
    BATCH_SIZE +
    ' --data ' +
    CONFIG_PATH +
    ' --img ' +
    IMG_SIZE +
    ' --cfg ' +
    MODEL_PATH +
    ' --epochs '+
    EPOCHS +
    ' --project ' +
    '/'.join(CONFIG_PATH.split("/")[:-1]) +
    ' --name exp'+
    " --weights ''")