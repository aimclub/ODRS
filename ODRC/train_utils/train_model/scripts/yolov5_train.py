import os
from pathlib import Path

def train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov5 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param GPU_COUNT: Number of video cards.
    """
    file = Path(__file__).resolve()
    os.system(
        f'OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node {GPU_COUNT} {file.parents[1]}/models/yolov5/train.py' +
        # ' --weights ' +
        # WEIGHTS + 
        ' --img ' + 
        IMG_SIZE +
        ' --batch ' +
        BATCH_SIZE +
        ' --epochs ' +
        EPOCHS +
        ' --data ' +
        CONFIG_PATH +
        ' --cfg ' +
        MODEL_PATH +
        ' --device '+
        SELECT_GPU +
        ' --project ' +
        '/'.join(CONFIG_PATH.split("/")[:-1]) +
        ' --name exp')
