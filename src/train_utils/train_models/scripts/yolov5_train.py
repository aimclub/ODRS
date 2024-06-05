import os
from pathlib import Path
import hashlib
from datetime import datetime


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
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    hash_value = int(hashlib.md5(now.encode()).hexdigest(), 16) % 100000

    command = "python3" if GPU_COUNT == 0 else f"OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node {GPU_COUNT} --master_port={hash_value:05d}"

    train_script_path = str(Path(file.parents[1]) / 'models' / 'yolov5' / 'train.py')

    full_command = (
        f"{command} {train_script_path}"
        f" --img {IMG_SIZE}"
        f" --batch {BATCH_SIZE}"
        f" --epochs {EPOCHS}"
        f" --data {CONFIG_PATH}"
        f" --cfg {MODEL_PATH}"
        f" --device {SELECT_GPU}"
        f" --project {CONFIG_PATH.parent}"
        f" --name exp"
    )
    os.system(full_command)
