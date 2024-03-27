from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
from loguru import logger
from pathlib import Path
import cv2
import numpy as np
file = Path(__file__).resolve()


def gini_coefficient(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_examples = len(labels)
    gini = 0
    for label in class_counts:
        label_prob = class_counts[label] / total_examples
        gini += label_prob * (1 - label_prob)
    return gini


def get_image_size(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        return width, height
    return None


def min_max_scaler(features):
    scaler = MinMaxScaler()
    features_normalized = np.exp(scaler.fit_transform(features))
    features_normalized /= np.sum(features_normalized, axis=0)
    return features_normalized


def get_average_fps(df, column, part_num):
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


def get_average_map50(df, column, part_num):
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
