import os
import subprocess


def train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov8 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param GPU_COUNT: Number of video cards.
    :param SELECT_GPU: GPU selection.
    """
    os.environ['WANDB_MODE'] = 'disabled'
    command = "yolo"

    full_command = (
        f"{command} detect train "
        f"data={CONFIG_PATH} "
        f"imgsz={IMG_SIZE} "
        f"batch={BATCH_SIZE} "
        f"epochs={EPOCHS} "
        f"model={MODEL_PATH} "
        f"device={SELECT_GPU} "
        f"name={CONFIG_PATH.parent}/exp"
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

# def train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
#     """
#     Runs yolov8 training using the parameters specified in the config.

#     :param IMG_SIZE: Size of input images as integer or w,h.
#     :param BATCH_SIZE: Batch size for training.
#     :param EPOCHS: Number of epochs to train for.
#     :param CONFIG_PATH: Path to config dataset.
#     :param MODEL_PATH: Path to model file (yaml).
#     :param GPU_COUNT: Number of video cards.
#     """
#     os.system(f"yolo detect train "
#               f"data={CONFIG_PATH} "
#               f"imgsz={IMG_SIZE} "
#               f"batch={BATCH_SIZE} "
#               f"epochs={EPOCHS} "
#               f"model={MODEL_PATH} "
#               f"device={SELECT_GPU} "
#               f"name={CONFIG_PATH.parent}/exp")
