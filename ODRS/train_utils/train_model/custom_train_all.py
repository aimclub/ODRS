import argparse
import os
import sys
from yaml import load
from yaml import FullLoader
from pathlib import Path
from scripts.yolov5_train import train_V5
from scripts.yolov7_train import train_V7
from scripts.yolov8_train import train_V8
from scripts.faster_rccn_train import train_frcnn
from scripts.ssd_train import train_ssd

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.data_utils.split_dataset import split_data, copy_arch_folder
from ODRS.data_utils.create_config import create_config_data, delete_cache
from ODRS.data_utils.convert_yolo_to_voc import convert_voc

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # PATH TO ODRC_project
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def load_config(config_file):
    # Returns loaded config for the chosen architecture
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


def fit_model(arch, DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH, CONFIG_PATH, SPLIT_TRAIN_VALUE,
              SPLIT_VAL_VALUE, SPLIT_TEST_VALUE, GPU_COUNT, SELECT_GPU):
    split_data(DATA_PATH, SPLIT_TRAIN_VALUE, SPLIT_VAL_VALUE, SPLIT_TEST_VALUE)

    if os.path.exists(f'{DATA_PATH}/train') and os.path.exists(f'{DATA_PATH}/valid'):
        PATH_SPLIT_TRAIN = f'{DATA_PATH}/train'
        PATH_SPLIT_VALID = f'{DATA_PATH}/valid'

    delete_cache(DATA_PATH)

    if arch == 'yolov8':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS)
        train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'yolov5':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS)
        train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'yolov7':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS)
        train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif arch == 'rcnn':
        DATA_PATH = copy_arch_folder(DATA_PATH)
        convert_voc(DATA_PATH, CLASSES)
        CONFIG_PATH = create_config_data(f'{DATA_PATH}/train', f'{DATA_PATH}/valid', CLASSES, CONFIG_PATH, arch,
                                         BATCH_SIZE, EPOCHS)
        train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, IMG_SIZE)
    elif arch == 'ssd':
        DATA_PATH = copy_arch_folder(DATA_PATH)
        convert_voc(DATA_PATH, CLASSES)
        CONFIG_PATH = create_config_data(f'{DATA_PATH}/train.json', f'{DATA_PATH}/valid.json', CLASSES, CONFIG_PATH,
                                         arch, BATCH_SIZE, EPOCHS)
        train_ssd(CONFIG_PATH)


def run(arch):
    """
    Create config, run learning functions.

    :param arhc: Trainable architecture.
    """
    config = load_config(f'{ROOT}/ODRS/train_utils/config/custom_config.yaml')

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

    fit_model(arch, DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH, CONFIG_PATH, SPLIT_TRAIN_VALUE,
              SPLIT_VAL_VALUE, SPLIT_TEST_VALUE, GPU_COUNT, SELECT_GPU)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='yolov8')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
