import os
from yaml import load
from yaml import FullLoader
from pathlib import Path
from datetime import datetime
from loguru import logger
import shutil
import sys

file = Path(__file__).resolve()

def get_classes_path(ROOT, classes_path):
    current_file_path = Path(__file__).resolve()
    DATA_PATH = Path(ROOT)
    CLASSES_PATH = Path(classes_path)
    try:
        if CLASSES_PATH.is_file():
            logger.info(f"Copying classes file to {DATA_PATH}")
            shutil.copy(classes_path, DATA_PATH)
    except Exception as e:
        logger.warning(f"An error has occurred: {e}")
    CLASSES_PATH = CLASSES_PATH.name
    return Path(current_file_path.parents[3]) / CLASSES_PATH


def load_class_names(classes_file):
    """ Загрузка названий классов из файла. """
    with open(classes_file, 'r') as file:
        class_names = [line.strip() for line in file]
    return class_names


def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)
    
    
def get_models():
    path_config = Path(file.parents[1]) / 'config_models' / 'models.yaml'
    config = load_config(path_config)
    models = config['models_array']
    return models


def create_run_directory(model):
    current_file_path = Path(__file__).resolve()

    runs_directory = Path(current_file_path.parents[3]) / 'runs'
    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory, exist_ok=True)

    runs_path = runs_directory / f"{str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}_{model}"
    os.makedirs(runs_path, exist_ok=True)
    return runs_path

def get_data_path(ROOT, folder_name):
    DATA_PATH = Path(ROOT) / 'user_datasets'
    FOLDER_PATH = DATA_PATH / folder_name
    target_path = DATA_PATH / FOLDER_PATH.name
    try:
        if not Path(FOLDER_PATH).is_dir() or not any(Path(FOLDER_PATH).iterdir()):
            logger.error("The dataset folder is empty or does not exist.")
            sys.exit(0)

        if os.path.isdir(target_path) and FOLDER_PATH.parent.resolve() != DATA_PATH.resolve():
            logger.error("The dataset folder is alredy exist.")
            sys.exit(0)

        if FOLDER_PATH.parent.resolve() != DATA_PATH.resolve():
            logger.info(f"Copying a set of images to {DATA_PATH}")
            shutil.copytree(FOLDER_PATH, target_path, dirs_exist_ok=True)
            FOLDER_PATH = target_path

    except Exception as e:
        logger.error(f"An error has occurred: {e}")
    return FOLDER_PATH