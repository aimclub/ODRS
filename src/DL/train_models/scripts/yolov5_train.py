import os
from pathlib import Path
import socket
import random
import subprocess

def get_free_port():
    while True:
        port = random.randint(0, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                # Если порт занят, попробуем снова
                continue

def train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov5 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param GPU_COUNT: Number of video cards.
    :param SELECT_GPU: GPU selection.
    """
    file = Path(__file__).resolve()
    os.environ['WANDB_MODE'] = 'disabled'
    
    command = "python3" if GPU_COUNT == 0 else f"OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node {GPU_COUNT} --master_port={get_free_port()}"

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

    # Logging the output to the console in real-time
    process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.decode().strip())

    rc = process.poll()
    return rc



# def train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
#     """
#     Runs yolov5 training using the parameters specified in the config.


#     :param IMG_SIZE: Size of input images as integer or w,h.
#     :param BATCH_SIZE: Batch size for training.
#     :param EPOCHS: Number of epochs to train for.
#     :param CONFIG_PATH: Path to config dataset.
#     :param MODEL_PATH: Path to model file (yaml).
#     :param GPU_COUNT: Number of video cards.
#     """
#     file = Path(__file__).resolve()

#     command = "python3" if GPU_COUNT == 0 else f"OMP_NUM_THREADS=1 python3 -m torch.distributed.run --nproc_per_node {GPU_COUNT} --master_port={get_free_port()}"

#     train_script_path = str(Path(file.parents[1]) / 'models' / 'yolov5' / 'train.py')

#     full_command = (
#         f"{command} {train_script_path}"
#         f" --img {IMG_SIZE}"
#         f" --batch {BATCH_SIZE}"
#         f" --epochs {EPOCHS}"
#         f" --data {CONFIG_PATH}"
#         f" --cfg {MODEL_PATH}"
#         f" --device {SELECT_GPU}"
#         f" --project {CONFIG_PATH.parent}"
#         f" --name exp"
#     )
#     os.system(full_command)
