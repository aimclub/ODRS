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


def prepare_to_train(config, list_parameters):
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

    if list_parameters:
        for i in range(len(list_parameters[list(list_parameters.keys())[0]])):
            current_params = {
                'DATA_PATH': DATA_PATH[i] if isinstance(DATA_PATH, list) else DATA_PATH,
                'CLASSES': CLASSES[i] if isinstance(CLASSES, list) else CLASSES,
                'IMG_SIZE': IMG_SIZE[i] if isinstance(IMG_SIZE, list) else IMG_SIZE,
                'BATCH_SIZE': BATCH_SIZE[i] if isinstance(BATCH_SIZE, list) else BATCH_SIZE,
                'EPOCHS': EPOCHS[i] if isinstance(EPOCHS, list) else EPOCHS,
                'MODEL': MODEL[i] if isinstance(MODEL, list) else MODEL,
                'CONFIG_PATH': CONFIG_PATH[i] if isinstance(CONFIG_PATH, list) else CONFIG_PATH,
                'SPLIT_TRAIN_VALUE': SPLIT_TRAIN_VALUE[i] if isinstance(SPLIT_TRAIN_VALUE, list) else SPLIT_TRAIN_VALUE,
                'SPLIT_VAL_VALUE': SPLIT_VAL_VALUE[i] if isinstance(SPLIT_VAL_VALUE, list) else SPLIT_VAL_VALUE,
                'GPU_COUNT': GPU_COUNT[i] if isinstance(GPU_COUNT, list) else GPU_COUNT,
                'SELECT_GPU': SELECT_GPU[i] if isinstance(SELECT_GPU, list) else SELECT_GPU
            }
            fit_model(**current_params)
            
    else:
        fit_model(DATA_PATH, CLASSES, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL, CONFIG_PATH, SPLIT_TRAIN_VALUE,
                SPLIT_VAL_VALUE, GPU_COUNT, SELECT_GPU)


def check_dict_arrays_sizes(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, list):
            first_array = next(iter(dictionary.values()))
            first_array_size = len(first_array)
            current_array_size = len(value)
            if current_array_size != first_array_size:
                raise ValueError(f"Size mismatch for key '{key}'. Expected size: {first_array_size}, actual size: {current_array_size}")


def run():
    config_path = Path(ROOT) / 'ODRS' / 'train_utils' / 'config' / 'custom_config.yaml'
    config = loadConfig(config_path)

    list_parameters = {key: value for key, value in config.items() if isinstance(value, list)}
    check_dict_arrays_sizes(list_parameters)
    prepare_to_train(config, list_parameters)


if __name__ == "__main__":
    run()
