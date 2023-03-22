import os

def train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, SAVE_PATH, GPU_COUNT):
    """
    Runs yolov5 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param SAVE_PATH: Path to save model.pt.
    :param GPU_COUNT:Number of video cards.
    """
    os.system(
        f'OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 5 yolov5/train.py --img ' +
        IMG_SIZE +
        ' --batch ' +
        BATCH_SIZE +
        ' --epochs ' +
        EPOCHS +
        ' --data ' +
        CONFIG_PATH +
        ' --cfg ' +
        MODEL_PATH +
        ' --weights ' +
        SAVE_PATH + 
        ' --device '+
        GPU_COUNT)