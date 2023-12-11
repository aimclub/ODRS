import os
import sys
from pathlib import Path
from loguru import logger
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.data_utils.split_dataset import split_data, copy_arch_folder
from ODRS.data_utils.resize_image import resize_images_and_annotations
from ODRS.data_utils.create_config import create_config_data, delete_cache
from ODRS.data_utils.convert_yolo_to_voc import convert_voc
from ODRS.train_utils.train_model.scripts.yolov5_train import train_V5
from ODRS.train_utils.train_model.scripts.yolov7_train import train_V7
from ODRS.train_utils.train_model.scripts.yolov8_train import train_V8
from ODRS.train_utils.train_model.scripts.faster_rccn_train import train_frcnn
from ODRS.train_utils.train_model.scripts.ssd_train import train_ssd
from ODRS.utils.utils import modelSelection, loadConfig, getDataPath, getClassesPath


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # PATH TO ODRS
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def fit_model(DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL, CONFIG_PATH, SPLIT_TRAIN_VALUE,
              SPLIT_VAL_VALUE, GPU_COUNT, SELECT_GPU):
    
    DATA_PATH = getDataPath(ROOT, DATA_PATH)
    CLASSES = getClassesPath(ROOT, CLASSES)

    PATH_SPLIT_TRAIN, PATH_SPLIT_VALID = split_data(DATA_PATH, SPLIT_TRAIN_VALUE, SPLIT_VAL_VALUE)

    arch, MODEL_PATH = modelSelection(MODEL)

    delete_cache(DATA_PATH)
    #ready
    if arch == 'yolov8':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS, MODEL)
        train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    #ready
    elif arch == 'yolov5':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS, MODEL)
        train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    #ready
    elif arch == 'yolov7':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_PATH, arch, BATCH_SIZE,
                                         EPOCHS, MODEL)
        train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    #ready
    elif arch == 'faster-rcnn':
        DATA_PATH = copy_arch_folder(DATA_PATH)
        resize_images_and_annotations(DATA_PATH, IMG_SIZE)
        convert_voc(DATA_PATH, CLASSES)
        CONFIG_PATH = create_config_data(Path(DATA_PATH) / 'train', Path(DATA_PATH) / 'valid', CLASSES, CONFIG_PATH, arch,
                                         BATCH_SIZE, EPOCHS, MODEL)
        train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, GPU_COUNT, IMG_SIZE)
    #ready
    elif arch == 'ssd':
        DATA_PATH = copy_arch_folder(DATA_PATH)
        resize_images_and_annotations(DATA_PATH, IMG_SIZE)
        convert_voc(DATA_PATH, CLASSES)
        CONFIG_PATH = create_config_data(Path(DATA_PATH) / 'train.json', Path(DATA_PATH) / 'valid.json', CLASSES, CONFIG_PATH,
                                         arch, BATCH_SIZE, EPOCHS, MODEL)
        train_ssd(CONFIG_PATH)


def run():
    """
    Create config, run learning functions.

    """
    config_path = Path(ROOT) / 'ODRS' / 'train_utils' / 'config' / 'custom_config.yaml'
    config = loadConfig(config_path)

    DATA_PATH = config['DATA_PATH']
    CLASSES = config['CLASSES']
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    MODEL = config['MODEL']
    CONFIG_PATH = config['CONFIG_PATH']
    SPLIT_TRAIN_VALUE = config['SPLIT_TRAIN_VALUE']
    SPLIT_VAL_VALUE = config['SPLIT_VAL_VALUE']
    GPU_COUNT = config['GPU_COUNT']
    SELECT_GPU = config['SELECT_GPU']

    fit_model(DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL, CONFIG_PATH, SPLIT_TRAIN_VALUE,
              SPLIT_VAL_VALUE, GPU_COUNT, SELECT_GPU)

if __name__ == "__main__":
    run()
