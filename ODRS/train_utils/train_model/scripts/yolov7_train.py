import os
from pathlib import Path


def train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov7 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as an integer or w, h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to the config dataset.
    :param MODEL_PATH: Path to the model file (yaml).
    """
    file = Path(__file__).resolve()

    command = "python3" if not GPU_COUNT else f"OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node {GPU_COUNT}"
    train_script_path = str(Path(file.parents[1]) / 'models' / 'yolov7' / 'train.py')
    full_command = (
        command +
        f" {train_script_path}" +
        f" --device {SELECT_GPU}" +
        f" --batch-size {BATCH_SIZE}" +
        f" --data {CONFIG_PATH}" +
        f" --img {IMG_SIZE}" +
        f" --cfg {MODEL_PATH}" +
        f" --epochs {EPOCHS}" +
        f" --project {CONFIG_PATH.parent}" +
        f" --name exp" +
        " --weights ''"
    )

    os.system(full_command)
