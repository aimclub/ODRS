from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import os
from loguru import logger
from pathlib import Path
import numpy as np
import csv
import yaml
from collections import Counter
file = Path(__file__).resolve()

from src.data_processing.ml_processing.plots import plot_class_balance
from src.data_processing.data_utils.utils import load_config, load_class_names
from src.data_processing.ml_processing.feature_perfomance import gini_coefficient, get_image_size
from src.data_processing.ml_processing.feature_perfomance import get_average_map50, get_average_fps, min_max_scaler


def load_yolo_labels(data_path, class_names):
    """ Загрузка меток классов из YOLO аннотаций. """
    dict_labels = dict()
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        labels = list()
        txt_folder = path / name / 'labels'
        for filename in os.listdir(txt_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(txt_folder, filename), 'r') as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            labels.append(class_names[class_id])
        dict_labels[name] = labels
    return dict_labels


def find_images(data_path):
    supported_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        for root, dirs, files in os.walk(path / name):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    image_paths.append(os.path.join(root, file))

    return image_paths


def dataset_info(dataset_path, classes_path, run_path):
    class_labels = list()
    class_names = load_class_names(classes_path)
    dict_class_labels = load_yolo_labels(dataset_path, class_names)
    for value in dict_class_labels.values():
        class_labels += value
    gini = "{:.2f}".format(gini_coefficient(class_labels))
    plot_class_balance(class_labels, run_path)

    dump_csv(class_names, class_labels, dict_class_labels, run_path)

    gini_coef = float(gini) * 100
    number_of_classes = len(set(class_labels))
    image_paths = find_images(dataset_path)
    img_w, img_h = get_image_size(image_paths[0])

    logger.info(f"Number of images: {len(image_paths)}")
    logger.info(f"Width: {img_w}")
    logger.info(f"Height: {img_h}")
    logger.info(f"Gini Coefficient: {gini_coef}")
    logger.info(f"Number of classes: {number_of_classes}")

    return [float(img_w), float(img_h), gini_coef,
            float(number_of_classes), len(image_paths)]



def get_config_data(path_config):
    config = load_config(path_config)
    mode = config['GPU']
    classes_path = config['classes_path']
    dataset_path = config['dataset_path']
    speed = config['speed']
    accuracy = config['accuracy']
    return mode, classes_path, dataset_path, speed, accuracy


def synthesize_data(df, num_samples):
    new_data = []
    for _ in range(num_samples):
        random_row = df.sample(n=1).iloc[0]
        # Варьируем данные с помощью некоторого случайного шума
        noise = np.random.normal(0, 0.1, df.shape[1])  # Среднее 0, стандартное отклонение 0.1
        new_row = random_row + noise
        new_data.append(new_row)
    return pd.DataFrame(new_data, columns=df.columns)


def get_data_rules(mode):
    data = pd.read_csv(Path(file.parents[1]) / 'data_rules' / 'model_cs.csv', delimiter=';')
    data = data.sample(frac=0.7, random_state=42)
    data = data.iloc[:, 0:9]
    numeric_columns = ['FPS_GPU', 'FPS_CPU']
    data[numeric_columns] = data[numeric_columns].replace(',', '', regex=True)
    if mode:
        data = data.drop('FPS_CPU', axis=1)
        data = data.rename(columns={'FPS_GPU': 'FPS'})
    else:
        data = data.drop('FPS_GPU', axis=1)
        data = data.rename(columns={'FPS_CPU': 'FPS'})
    data = data.astype(float)
    return data


def data_processing(dataset_data, mode, speed, accuracy):
    data = get_data_rules(mode)
    speed = get_average_fps(data, 'FPS', speed)
    accuracy = get_average_map50(data, 'mAP50', accuracy)
    if accuracy is None or speed is None:
        print("Invalid part number!")
    else:
        dataset_data.append(speed)
        dataset_data.append(accuracy)

        warnings.filterwarnings("ignore")
        data_add = data.iloc[:, 0:7]
        data_add = data_add.append(pd.Series(dataset_data, index=data_add.columns), ignore_index=True)

        features = data_add[data_add.columns[0:7]].values
        labels = data['Model'].values

        features_normalized = min_max_scaler(features)

        dataset_data = features_normalized[-1]
        features_normalized = features_normalized[:-1]
        return features_normalized, labels


def dump_csv(class_names, class_labels, dict_class_labels, run_path):
    for key, value in dict_class_labels.items():
        dict_class_labels[key] = Counter(value)
    dict_class_labels['all'] = Counter(class_labels)

    for key, value in dict_class_labels.items():
        for class_name in class_names:
            if class_name not in value.keys():
                value.update({f'{class_name}': 0})
    csv_file_path = Path(run_path) / 'class_counts.csv'
    file_exists = csv_file_path.is_file()

    with open(csv_file_path, 'a', newline='') as csvfile:
        field_names = ['class-name']
        for key in dict_class_labels:
            field_names.append(f'{key}-count')
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        if not file_exists:
            writer.writeheader()
        all_values = dict()
        for class_name in class_names:
            values = list()
            for class_value in dict_class_labels.values():
                for key, value in class_value.items():
                    if key == class_name:
                        values.append(value)
            all_values[class_name] = values

        sorted_dict = reversed(sorted(dict_class_labels['all'].items(), key=lambda x: x[1]))

        for class_key, class_value in sorted_dict:
            for key, value in all_values.items():
                if key == class_key:
                    if len(field_names) == 5:
                        writer.writerow({
                            field_names[0]: key,
                            field_names[1]: value[0],
                            field_names[2]: value[1],
                            field_names[3]: value[2],
                            field_names[4]: value[3]
                            })
                    if len(field_names) == 4:
                        writer.writerow({
                            field_names[0]: key,
                            field_names[1]: value[0],
                            field_names[2]: value[1],
                            field_names[3]: value[2]
                            })
                    if len(field_names) == 3:
                        writer.writerow({field_names[0]: key, field_names[1]: value[0], field_names[2]: value[1]})


def dump_yaml(mode, classes_path, dataset_path, speed, accuracy, dataset_data, model_top, run_path):
    data = {'GPU': mode,
            'accuracy': accuracy,
            'classes_path': classes_path,
            'dataset_path': dataset_path,
            'speed': speed,
            'Number_of_images': dataset_data[4],
            'image_Width': dataset_data[0],
            'image_Height': dataset_data[1],
            'Gini_Coefficient': dataset_data[2],
            'Number_of_classes': dataset_data[3],
            'Top_1': model_top[0],
            'Top_2': model_top[1],
            'Top_3': model_top[2]
            }
    with open(run_path / 'results.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)