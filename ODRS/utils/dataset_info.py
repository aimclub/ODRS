import cv2
import os
import numpy as np
from loguru import logger
from pathlib import Path
from collections import Counter
from ODRS.utils.ml_plot import plot_class_balance
from ODRS.utils.ml_utils import dumpCSV

def load_class_names(classes_file):
    """ Загрузка названий классов из файла. """
    with open(classes_file, 'r') as file:
        class_names = [line.strip() for line in file]
    return class_names


def load_yolo_labels(data_path, class_names):
    """ Загрузка меток классов из YOLO аннотаций. """
    labels = []
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        txt_folder = path / name / 'labels'
        for filename in os.listdir(txt_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(txt_folder, filename), 'r') as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            labels.append(class_names[class_id])
    return labels


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



def dataset_info(dataset_path, classes_path, run_path):
    class_names = load_class_names(classes_path)
    class_labels = load_yolo_labels(dataset_path, class_names)
    gini = "{:.2f}".format(gini_coefficient(class_labels))
    plot_class_balance(class_labels, run_path)
    
    dumpCSV(class_names, class_labels, run_path)


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
