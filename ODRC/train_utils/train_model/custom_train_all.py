import argparse
import yaml
import shutil
import os
import subprocess
import contextlib

import sys

from tqdm import tqdm
from loguru import logger
from yaml import load
from yaml import FullLoader

from scripts.yolov5_train import train_V5
from scripts.yolov7_train import train_V7
from scripts.yolov8_train import train_V8
from scripts.ssd_train import train_ssd
from scripts.faster_rcnn_train import train_frcnn

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3] #PATH TO ODRC_project
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ODRC.data_utils.split_dataset import split_data
from ODRC.data_utils.create_config_yolo import create_config_data
from ODRC.data_utils.create_config_yolo import delete_cache
#from data_utils.split_dataset import split_data



def load_config(config_file):
    # Returns loaded config сhosen architecture
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


def run(arch, split):
    """
    Create config, run learning functions and split dataset if you need it.  

    :param arhc: Trainable architecture.
    :param split: Creating train and validation dataset.
    """
    if arch == "yolov5":
        config = load_config('config/custom_config_v5.yaml')
    elif arch == "yolov8":
        config = load_config('config/custom_config_v8.yaml')
    elif arch == "yolov7":
        config = load_config('config/custom_config_v7.yaml')
    elif arch == "rcnn":
        config = load_config('config/custom_config_rcnn.yaml')
    else:
        config = load_config('config/custom_config_ssd.yaml')

    DATASET = config['DATASET']
    WEIGHTS = config['WEIGHTS']
    DATA_PATH = config['DATA_PATH']
    CLASSES = config['CLASSES']
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    MODEL_PATH = config['MODEL_PATH']
    CONFIG_PATH = config['CONFIG_PATH']
    SPLIT_TRAIN_VALUE = config['SPLIT_TRAIN_VALUE']
    SPLIT_VAL_VALUE = config['SPLIT_VAL_VALUE']
    SPLIT_TEST_VALUE = config['SPLIT_TEST_VALUE']
    GPU_COUNT = config['GPU_COUNT']
    SELECT_GPU = config['SELECT_GPU']


    if split:
        PATH_SPLIT_TRAIN, PATH_SPLIT_VALID =  split_data(DATA_PATH, SPLIT_TRAIN_VALUE, SPLIT_VAL_VALUE, SPLIT_TEST_VALUE)
    elif os.path.exists(f'{DATA_PATH}/train') and os.path.exists(f'{DATA_PATH}/val'):
        PATH_SPLIT_TRAIN = f'{DATA_PATH}/train'
        PATH_SPLIT_VALID = f'{DATA_PATH}/val'
        print("Directory with split dataset already exist")
    else:
        print("Create train and val folders in your dataset folder or set the SPLIT parameter to True")
        sys.exit()

    delete_cache(DATA_PATH)

    CONFIG_PATH =  create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch)

    if arch == 'yolov8': #READY
        train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'yolov5': #READY
        train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'yolov7': #READY
        train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'rcnn': #READY
        train_frcnn(BATCH_SIZE, GPU_COUNT, SELECT_GPU, DATA_PATH)
    elif arch == 'ssd': #READY
        train_ssd(BATCH_SIZE, CONFIG_PATH, DATA_PATH, GPU_COUNT, DATASET)
    

def parse_opt():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='yolov8')
    parser.add_argument('--split', type=bool, default=False)
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))