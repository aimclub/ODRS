import os

def train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU):
    """
    Runs yolov8 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param GPU_COUNT:Number of video cards.
    """

    os.system(f'yolo detect train data=' +
    CONFIG_PATH +
    ' imgsz=' +
    IMG_SIZE +
    ' batch=' +
    BATCH_SIZE +
    ' epochs=' +
    EPOCHS +
    ' model='+
    MODEL_PATH +
    ' device=' +
    SELECT_GPU +
    ' project=' +
    '/'.join(CONFIG_PATH.split("/")[:-1]) +
    ' name=exp'
    )