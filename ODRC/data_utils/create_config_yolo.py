from loguru import logger
from yaml import load
from yaml import FullLoader
import yaml
from pathlib import Path
import os
from datetime import datetime

def create_class_list(filename):
    #Returns list of classes
    with open(filename, "r") as file_object:
        class_list = file_object.read().splitlines()
    return class_list


def delete_cache(data_path):
    extensions_to_delete = ['labels.cache', 'train.cache', 'val.cache']
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions_to_delete)):
                os.remove(os.path.join(root, file))


def create_config_data(train_path, val_path, classname_file, config_path, arch):
     # Get current file path
    current_file_path = Path(__file__).resolve()

    # Create runs directory if does not exists
    runs_directory = f"{current_file_path.parents[2]}/runs/"
    if not os.path.exists(runs_directory):
        os.makedirs(runs_directory, exist_ok=True)

    # Create runs path
    runs_path = f"{runs_directory}/{str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}_{arch}"
    os.makedirs(f"{runs_path}", exist_ok=True)

    # Create config path
    config_path = f"{runs_path}/{config_path}"

    # Create data config
    class_file_path = f"{current_file_path.parents[2]}/{classname_file}"
    class_list = create_class_list(class_file_path)
    data = dict(
        train=train_path,
        val=val_path,
        nc=len(class_list),
        names=class_list
    )

    # Create config file
    logger.info("Create config file")
    with open(config_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

    return config_path
