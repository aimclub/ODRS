import os

def train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT):
    """
    Runs yolov7 training using the parameters specified in the config.

    :param IMG_SIZE: Size of input images as integer or w,h.
    :param BATCH_SIZE: Batch size for training.
    :param EPOCHS: Number of epochs to train for.
    :param CONFIG_PATH: Path to config dataset.
    :param MODEL_PATH: Path to model file (yaml).
    :param GPU_COUNT:Number of video cards.
    """
    os.system('OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node 5 yolov7/train.py'
    ' --device ' +
    GPU_COUNT +
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
    ' --name run'+
    " --weights ''")