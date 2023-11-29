from loguru import logger
import yaml
from pathlib import Path
import os
from datetime import datetime
from ODRS.data_utils.prepare_ssd import read_names_from_txt


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
