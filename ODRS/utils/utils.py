from pathlib import Path
from loguru import logger
from yaml import load
from yaml import FullLoader
import shutil
import sys
import os


file = Path(__file__).resolve()


def loadConfig(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)
    

def get_models():
    path_config = Path(file.parents[1]) / 'config_models' / 'models.yaml'
    config = loadConfig(path_config)
    models = config['models_array']
    return models


def modelSelection(MODEL):
    arch = ""
    if MODEL.startswith('yolov5'):
        arch = 'yolov5'
        path_config = Path(file.parents[1]) / 'train_utils'/ 'train_model' / 'models' / 'yolov5' / 'models' / f'{MODEL}.yaml'
        if os.path.exists(path_config):
            return arch, path_config
        else:
            logger.error("There is no such model in our database")
            sys.exit()

    elif MODEL.startswith('yolov7'):
        arch = 'yolov7'
        path_config = Path(file.parents[1]) / 'train_utils' / 'train_model' / 'models' / 'yolov7' / 'cfg' / 'training' / f'{MODEL}.yaml'
        if os.path.exists(path_config):
            return arch, path_config
        else:
            logger.error("There is no such model in our database")
            sys.exit()
    
    elif MODEL.startswith('yolov8'):
        arch = 'yolov8'
        path_config = Path(file.parents[1]) / 'train_utils' / 'train_model' / 'models' / 'ultralytics' / 'ultralytics' / 'models' / 'v8' / f'{MODEL}.yaml'
        if os.path.exists(path_config):
            return arch, path_config
        else:
            logger.error("There is no such model in our database")
            sys.exit()
    
    elif MODEL == 'ssd':
        arch = 'ssd'
        return arch, None
    
    elif MODEL == 'faster-rcnn':
        arch = 'faster-rcnn'
        return arch, None
    
    else:
        logger.critical("Invalid model name. ModelSelection")


def getDataPath(ROOT, folder_name):
    DATA_PATH = Path(ROOT) / 'user_datasets'
    FOLDER_PATH = DATA_PATH / folder_name
    try:
        if not Path(FOLDER_PATH).is_dir() or not any(Path(FOLDER_PATH).iterdir()):
            logger.error("The dataset folder is empty or does not exist.")
            sys.exit(0)
            return

        if FOLDER_PATH.parent.resolve() != DATA_PATH.resolve():
            target_path = DATA_PATH / FOLDER_PATH.name
            logger.info(f"Copying a set of images to {DATA_PATH}")
            shutil.copytree(FOLDER_PATH, target_path, dirs_exist_ok=True)
            FOLDER_PATH = target_path
    
    except Exception as e:
        logger.error(f"An error has occurred: {e}")
    return FOLDER_PATH


def getClassesPath(ROOT, classes_path):
    DATA_PATH = Path(ROOT)
    CLASSES_PATH = Path(classes_path)
    try:
        if CLASSES_PATH.is_file():
            logger.info(f"Copying classes file to {DATA_PATH}")
            shutil.copy(classes_path, DATA_PATH)
    
    except Exception as e:
        logger.warning(f"An error has occurred: {e}")
    CLASSES_PATH = CLASSES_PATH.name

    return CLASSES_PATH
