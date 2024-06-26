import os
import sys
from pathlib import Path
from loguru import logger
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from src.data_processing.data_utils.utils import load_config, get_data_path, get_classes_path
from src.data_processing.data_utils.split_dataset import split_data, copy_arch_folder, resize_images_and_annotations
from src.data_processing.train_processing.prepare_train import  model_selection, delete_cache
from src.data_processing.train_processing.prepare_train import create_config_data, check_config_arrays_sizes
from src.data_processing.train_processing.convert_yolo_to_voc import convert_voc

from src.DL.train_models.scripts import yolov8_train, yolov7_train, yolov5_train
from src.DL.train_models.scripts import faster_rccn_train, ssd_train




FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # PATH TO ODRS
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def fit_model(DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL, SPLIT_TRAIN_VALUE,
              SPLIT_VAL_VALUE, GPU_COUNT, SELECT_GPU, CONFIG_NAME = 'dataset.yaml'):
    
    DATA_PATH = get_data_path(ROOT, DATA_PATH)
    CLASSES_PATH = get_classes_path(ROOT, CLASSES)

    PATH_SPLIT_TRAIN, PATH_SPLIT_VALID = split_data(DATA_PATH, SPLIT_TRAIN_VALUE, SPLIT_VAL_VALUE)

    ARCH, MODEL_PATH = model_selection(MODEL)

    delete_cache(DATA_PATH)

    if ARCH == 'yolov8':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES_PATH, CONFIG_NAME, ARCH, BATCH_SIZE,
                                         EPOCHS, MODEL)
        yolov8_train.train_V8(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif ARCH == 'yolov5':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES_PATH, CONFIG_NAME, ARCH, BATCH_SIZE,
                                         EPOCHS, MODEL)
        yolov5_train.train_V5(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif ARCH == 'yolov7':
        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES_PATH, CONFIG_NAME, ARCH, BATCH_SIZE,
                                         EPOCHS, MODEL)
        yolov7_train.train_V7(IMG_SIZE, BATCH_SIZE, EPOCHS, CONFIG_PATH, MODEL_PATH, GPU_COUNT, SELECT_GPU)
    elif ARCH == 'faster-rcnn':

        DATA_PATH = copy_arch_folder(DATA_PATH)
        PATH_SPLIT_TRAIN = Path(DATA_PATH) / 'train'
        PATH_SPLIT_VALID = Path(DATA_PATH) / 'valid'
        resize_images_and_annotations(DATA_PATH, IMG_SIZE)
        convert_voc(DATA_PATH, CLASSES)

        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES_PATH, CONFIG_NAME, ARCH,
                                         BATCH_SIZE, EPOCHS, MODEL)
        faster_rccn_train.train_frcnn(CONFIG_PATH, EPOCHS, BATCH_SIZE, GPU_COUNT, IMG_SIZE)
    elif ARCH == 'ssd':

        DATA_PATH = copy_arch_folder(DATA_PATH)
        PATH_SPLIT_TRAIN = Path(DATA_PATH) / 'train.json'
        PATH_SPLIT_VALID = Path(DATA_PATH) / 'valid.json'
        resize_images_and_annotations(DATA_PATH, IMG_SIZE)
        convert_voc(DATA_PATH, CLASSES)

        CONFIG_PATH = create_config_data(PATH_SPLIT_TRAIN, PATH_SPLIT_VALID, CLASSES, CONFIG_NAME,
                                         ARCH, BATCH_SIZE, EPOCHS, MODEL)
        ssd_train.train_ssd(CONFIG_PATH)


def prepare_to_train(config, list_parameters):
    DATA_PATH = config['DATA_PATH']
    CLASSES = config['CLASSES']
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    MODEL = config['MODEL']
    CONFIG_NAME = 'dataset.yaml'
    SPLIT_TRAIN_VALUE = config['SPLIT_TRAIN_VALUE']
    SPLIT_VAL_VALUE = config['SPLIT_VAL_VALUE']
    GPU_COUNT = config['GPU_COUNT']
    SELECT_GPU = config['SELECT_GPU']

    if list_parameters:
        for i in range(len(list_parameters[list(list_parameters.keys())[0]])):
            current_params = {
                'DATA_PATH': DATA_PATH[i] if isinstance(DATA_PATH, list) else DATA_PATH,
                'CLASSES': CLASSES[i] if isinstance(CLASSES, list) else CLASSES,
                'IMG_SIZE': IMG_SIZE[i] if isinstance(IMG_SIZE, list) else IMG_SIZE,
                'BATCH_SIZE': BATCH_SIZE[i] if isinstance(BATCH_SIZE, list) else BATCH_SIZE,
                'EPOCHS': EPOCHS[i] if isinstance(EPOCHS, list) else EPOCHS,
                'MODEL': MODEL[i] if isinstance(MODEL, list) else MODEL,
                'CONFIG_NAME': CONFIG_NAME[i] if isinstance(CONFIG_NAME, list) else CONFIG_NAME,
                'SPLIT_TRAIN_VALUE': SPLIT_TRAIN_VALUE[i] if isinstance(SPLIT_TRAIN_VALUE, list) else SPLIT_TRAIN_VALUE,
                'SPLIT_VAL_VALUE': SPLIT_VAL_VALUE[i] if isinstance(SPLIT_VAL_VALUE, list) else SPLIT_VAL_VALUE,
                'GPU_COUNT': GPU_COUNT[i] if isinstance(GPU_COUNT, list) else GPU_COUNT,
                'SELECT_GPU': SELECT_GPU[i] if isinstance(SELECT_GPU, list) else SELECT_GPU
            }
            fit_model(**current_params)
            
    else:
        fit_model(DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL, SPLIT_TRAIN_VALUE,
                SPLIT_VAL_VALUE, GPU_COUNT, SELECT_GPU, CONFIG_NAME)


def run():
    config_path = Path(ROOT) / 'src' / 'DL' / 'config' / 'train_config.yaml'
    config = load_config(config_path)

    list_parameters = {key: value for key, value in config.items() if isinstance(value, list)}
    check_config_arrays_sizes(list_parameters)
    prepare_to_train(config, list_parameters)


if __name__ == "__main__":
    run()
