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
from ODRS.data_utils.create_config import createRunDirectory
from ODRS.utils.dataset_info import dataset_info
from ODRS.utils.utils import getDataPath
from ODRS.data_utils.split_dataset import split_data
from ODRS.utils.ml_utils import getModels, getConfigData, dataProcessing, dumpYAML

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # PATH TO ODRS
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



def predict(mode, classes_path, dataset_path, speed, accuracy):
    file = Path(__file__).resolve()

    #split_data(dataset_path, split_train_value=0.6, split_valid_value=0.35)

    run_path = createRunDirectory(model='ml')

    model_top = list()

    model_array = getModels()

    dataset_path_new = getDataPath(ROOT, dataset_path)

    dataset_data = dataset_info(dataset_path_new, Path(file.parents[2]) / classes_path, run_path)

    features_normalized, labels = dataProcessing(dataset_data, mode, speed, accuracy)

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

    dumpYAML(mode, classes_path, dataset_path, speed, accuracy, dataset_data, model_top, run_path)


def ml_main():
    file = Path(__file__).resolve()
    mode, classes_path, dataset_path, speed, accuracy = getConfigData(Path(file.parents[0]) / 'config' / 'ml_config.yaml')
    predict(mode, classes_path, dataset_path, speed, accuracy)

if __name__ == "__main__":
    ml_main()
