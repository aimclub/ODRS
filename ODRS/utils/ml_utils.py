from sklearn.preprocessing import MinMaxScaler
from ODRS.utils.utils import loadConfig
import csv
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import yaml
file = Path(__file__).resolve()

def getAverageFPS(df, column, part_num):
    sorted_fps = np.sort(df[column])
    num_parts = 5
    part_size = len(sorted_fps) // num_parts
    if part_num < 1 or part_num > num_parts:
        return None
    start_idx = (num_parts - part_num) * part_size
    end_idx = (num_parts - part_num + 1) * part_size
    selected_values = sorted_fps[start_idx:end_idx]
    average_fps = np.mean(selected_values)
    return average_fps


def getAverage_mAP50(df, column, part_num):
    sorted_mAP50 = np.sort(df[column])
    num_parts = 10
    part_size = len(sorted_mAP50) // num_parts

    if part_num < 1 or part_num > num_parts:
        return None

    start_idx = (part_num - 1) * part_size
    end_idx = part_num * part_size if part_num < num_parts else len(sorted_mAP50)
    selected_values = sorted_mAP50[start_idx:end_idx]
    average_mAP50 = np.mean(selected_values)
    return average_mAP50


def getConfigData(path_config):
    config = loadConfig(path_config)
    mode = config['GPU']
    classes_path = config['classes_path']
    dataset_path = config['dataset_path']
    speed = config['speed']
    accuracy = config['accuracy']
    return mode, classes_path, dataset_path, speed, accuracy


def getModels():
    path_config = Path(file.parent) / 'config_models' / 'models.yaml'
    config = loadConfig(path_config)
    models = config['models_array']
    return models


def synthesize_data(df, num_samples):
    new_data = []
    for _ in range(num_samples):
        random_row = df.sample(n=1).iloc[0]
        # Варьируем данные с помощью некоторого случайного шума
        noise = np.random.normal(0, 0.1, df.shape[1])  # Среднее 0, стандартное отклонение 0.1
        new_row = random_row + noise
        new_data.append(new_row)
    return pd.DataFrame(new_data, columns=df.columns)


def min_max_scaler(features):
    scaler = MinMaxScaler()
    features_normalized = np.exp(scaler.fit_transform(features))
    features_normalized /= np.sum(features_normalized, axis=0)
    return features_normalized


def getData(mode):
    data = pd.read_csv(Path(file.parents[0]) / 'data_train_ml' / 'model_cs.csv', delimiter=';')
    data = data.sample(frac=0.7, random_state=42) #frac = 0.7
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


def dumpCSV(class_names, class_labels, dict_class_labels, run_path):
    for key, value in dict_class_labels.items():
        dict_class_labels[key] = Counter(value)
    dict_class_labels['all'] = Counter(class_labels)

    for key, value in dict_class_labels.items():
        for class_name in class_names:
            if class_name not in value.keys():
                value.update({f'{class_name}': 0})
    csv_file_path = run_path / 'class_counts.csv'
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
                        writer.writerow({field_names[0]: key, field_names[1]: value[0], field_names[2]: value[1], field_names[3]: value[2], field_names[4]: value[3]})
                    if len(field_names) == 4:
                        writer.writerow({field_names[0]: key, field_names[1]: value[0], field_names[2]: value[1], field_names[3]: value[2]})
                    if len(field_names) == 3:
                        writer.writerow({field_names[0]: key, field_names[1]: value[0], field_names[2]: value[1]})


def dumpYAML(mode, classes_path, dataset_path, speed, accuracy, dataset_data, model_top, run_path):
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


def dataProcessing(dataset_data, mode, speed, accuracy):
    data = getData(mode)
    speed = getAverageFPS(data, 'FPS', speed)
    accuracy = getAverage_mAP50(data, 'mAP50', accuracy)
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
