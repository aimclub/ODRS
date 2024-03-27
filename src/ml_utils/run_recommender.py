import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import sys
import os
from pathlib import Path
from loguru import logger

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from src.data_processing.data_utils.utils import get_models, create_run_directory, get_data_path
from src.data_processing.data_utils.split_dataset import split_data
from src.data_processing.ml_processing.info_processor import get_config_data, dataset_info, data_processing, dump_yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # PATH TO ODRS
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



def predict(mode, classes_path, dataset_path, speed, accuracy):
    file = Path(__file__).resolve()

    model_top = list()

    run_path = create_run_directory(model='ml')
    model_array = get_models()
    dataset_path_new = get_data_path(ROOT, dataset_path)

    split_data(dataset_path_new, split_train_value=0.75, split_valid_value=0.15)

    dataset_data = dataset_info(dataset_path_new, Path(file.parents[2]) / classes_path, run_path)

    features_normalized, labels = data_processing(dataset_data, mode, speed, accuracy)

    random_forest = RandomForestClassifier(criterion='gini',
                                            min_samples_leaf=3, max_depth=25, n_estimators=52, random_state=42)
    ovrc = OneVsRestClassifier(random_forest)
    ovrc.fit(features_normalized, labels)

    y_pred = ovrc.predict(features_normalized)
    #accuracy_sc = accuracy_score(labels, y_pred)

    probabilities = ovrc.predict_proba([dataset_data])

    top_3_models = np.argsort(probabilities, axis=1)[:, ::-1][:, :3]

    logger.info("Top models for training:")
    for num_model in range(len(top_3_models[0])):
        model = model_array[top_3_models[0][int(num_model)]]
        model_top.append(model)
        logger.info(f'{num_model + 1}) {model}')

    dump_yaml(mode, classes_path, dataset_path, speed, accuracy, dataset_data, model_top, run_path)


def ml_main():
    file = Path(__file__).resolve()
    mode, classes_path, dataset_path, speed, accuracy = get_config_data(Path(file.parents[0]) / 'config' / 'ml_config.yaml')
    predict(mode, classes_path, dataset_path, speed, accuracy)

if __name__ == "__main__":
    ml_main()
