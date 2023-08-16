from yaml import load
from yaml import FullLoader
from pathlib import Path
from loguru import logger


file = Path(__file__).resolve()


def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


def get_models():
    path_config = f'{file.parents[0]}/config_models/models.yaml'
    config = load_config(path_config)
    models = config['models_array']
    return models


def get_path_model(name_model):
    if name_model.startswith('yolov5'):
        return f'{file.parents[1]}/train_utils/train_model/models/yolov5/models/{name_model}.yaml'
    elif name_model.startswith('yolov7'):
        return f'{file.parents[1]}/train_utils/train_model/models/yolov7/cfg/training/{name_model}.yaml'
    elif name_model.startswith('yolov8'):
        return f'{file.parents[1]}/train_utils/train_model/models/ultralytics/ultralytics/models/v8/{name_model}.yaml'
    elif name_model == 'ssd':
        return None
    elif name_model == 'faster-rcnn':
        return None
    else:
        logger.critical("Invalid model name")
