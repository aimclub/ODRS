from pathlib import Path
from loguru import logger
from yaml import load
from yaml import FullLoader
import shutil
import sys
import os
from loguru import logger
import yaml
from pathlib import Path
import os
from datetime import datetime
from src.data_processing.prepare_ssd import read_names_from_txt


file = Path(__file__).resolve()


def load_config(config_file):
    with open(config_file) as f:
        return load(f, Loader=FullLoader)


def get_models():
    path_config = Path(file.parents[1]) / 'config_models' / 'models.yaml'
    config = load_config(path_config)
    models = config['models_array']
    return models


def model_selection(MODEL):
    arch = ""
    if MODEL.startswith('yolov5'):
        arch = 'yolov5'
        path_config = Path(file.parents[1]) / 'train_utils' / 'train_model' / 'models' / 'yolov5' / 'models' / f'{MODEL}.yaml'
        if os.path.exists(path_config):
            return arch, path_config
        else:
            logger.error("There is no such model in our database")
            sys.exit()

    elif MODEL.startswith('yolov7'):
        arch = 'yolov7'
        path_config = (
            Path(file.parents[1]) / 'train_utils' / 'train_model' / 'models' /
            'yolov7' / 'cfg' / 'training' / f'{MODEL}.yaml'
            )
        if os.path.exists(path_config):
            return arch, path_config
        else:
            logger.error("There is no such model in our database")
            sys.exit()

    elif MODEL.startswith('yolov8'):
        arch = 'yolov8'
        path_config = (
            Path(file.parents[1]) / 'train_utils' / 'train_model' / 'models' /
            'ultralytics' / 'ultralytics' / 'models' / 'v8' / f'{MODEL}.yaml'
            )
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


def get_data_path(ROOT, folder_name):
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


def get_classes_path(ROOT, classes_path):
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

#-------------------------------------------------CREATE CONFIG
def create_class_list(filename):
    # Returns list of classes
    with open(filename, "r") as file_object:
        class_list = file_object.read().splitlines()
    return class_list


def delete_cache(data_path):
    extensions_to_delete = ['labels.cache', 'train.cache', 'val.cache']
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions_to_delete)):
                os.remove(os.path.join(root, file))


def createRunDirectory(model):
    current_file_path = Path(__file__).resolve()

    runs_directory = Path(current_file_path.parents[2]) / 'runs'
    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory, exist_ok=True)

    runs_path = runs_directory / f"{str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}_{model}"
    os.makedirs(runs_path, exist_ok=True)
    return runs_path


def create_config_data(train_path, val_path, classname_file, config_path, arch, batch_size, epochs, model):
    current_file_path = Path(__file__).resolve()

    runs_path = createRunDirectory(model)
    class_file_path = Path(current_file_path.parents[2]) / classname_file

    config_path = runs_path / config_path
    if arch == 'ssd':
        class_names = read_names_from_txt(class_file_path)
        dataset_yaml = '''\
# Data
train_json: {}
val_json: {}
class_names: {}
recall_steps: 11
image_mean: [123., 117., 104.]
image_stddev: [1., 1, 1.]

# Model
model: SSD
backbone:
  name: VGG16
  num_stages: 6
input_size: 300
anchor_scales: [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
anchor_aspect_ratios: [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]

# Training
batch_size: {}
epochs: {}
optim:
  name: SGD
  lr: 0.0001
  momentum: 0.9
  weight_decay: 0.0005
scheduler:
  name: MultiStepLR
  milestones: [155, 195]
  gamma: 0.1
            '''.format(train_path, val_path, class_names, batch_size, epochs)
        logger.info("Create config file")
        with open(config_path, 'w') as file:
            file.write(dataset_yaml)

        return config_path

    elif arch == 'faster-rcnn':
        classes = read_names_from_txt(class_file_path)
        class_names = ['__background__']
        for name in classes:
            class_names.append(name)

        dataset_yaml = '''\
# Images and labels directory should be relative to train.py
TRAIN_DIR_IMAGES: {}
TRAIN_DIR_LABELS: {}
# VALID_DIR should be relative to train.py
VALID_DIR_IMAGES: {}
VALID_DIR_LABELS: {}

# Class names.
CLASSES: {}

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC: {}

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
            '''.format(train_path / 'images', train_path / 'annotations', val_path / 'images',
                       val_path / 'annotations', class_names, len(class_names))
        logger.info("Create config file")
        with open(config_path, 'w') as file:
            file.write(dataset_yaml)

        return config_path

    else:
        class_list = create_class_list(class_file_path)
        data = dict(
            train=train_path,
            val=val_path,
            nc=len(class_list),
            names=class_list
        )
        logger.info("Create config file")
        with open(config_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        return config_path

