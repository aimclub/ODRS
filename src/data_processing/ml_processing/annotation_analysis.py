import pandas as pd
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from src.data_processing.data_utils.utils import load_class_names
from src.data_processing.ml_processing.plots import plot_class_balance
import numpy as np
import os
import cv2
import csv

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


def calculate_iou(bbox1, bbox2):
    """
    Вычисляет Intersection over Union (IoU) для двух ограничивающих прямоугольников.
    Каждый bbox задается как [x_min, y_min, x_max, y_max].
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    try:
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    except:
        iou = intersection_area
    return iou


def analysis_yolo_annotations(annotation_paths):
    bbox_sizes = []
    aspect_ratios = []
    objects_per_image = defaultdict(int)
    overlaps = []

    for annotation_path in tqdm(annotation_paths, desc="Annotation analyze"):
            image_id = annotation_path.split('.')[0]
            bboxes = []
            with open(os.path.join(annotation_path), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, x_center, y_center, width, height = map(float, parts)
                    bboxes.append([x_center - width / 2, y_center - height / 2,
                                   x_center + width / 2, y_center + height / 2])
                    bbox_sizes.append((width, height))
                    try:
                        aspect_ratios.append(width / height)
                    except:
                        aspect_ratios.append(width)
                    objects_per_image[image_id] += 1

            # Анализ перекрытий
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    iou = calculate_iou(bboxes[i], bboxes[j])
                    if iou > 0:
                        overlaps.append(iou)
    try:
        avg_objects_per_image = sum(objects_per_image.values()) / len(objects_per_image)
    except:
        avg_objects_per_image = 1
    bbox_sizes_df = pd.DataFrame(bbox_sizes, columns=['Width', 'Height'])
    aspect_ratios_df = pd.DataFrame(aspect_ratios, columns=['Aspect Ratio'])

    analysis_results = {
        'Average BBox Width': bbox_sizes_df['Width'].mean(),
        'Average BBox Height': bbox_sizes_df['Height'].mean(),
        'Min BBox Width': bbox_sizes_df['Width'].min(),
        'Min BBox Height': bbox_sizes_df['Height'].min(),
        'Max BBox Width': bbox_sizes_df['Width'].max(),
        'Max BBox Height': bbox_sizes_df['Height'].max(),
        'Average Aspect Ratio': aspect_ratios_df['Aspect Ratio'].mean(),
        'Average Objects Per Image': avg_objects_per_image,
        'Average Overlap': sum(overlaps) / len(overlaps) if overlaps else 0,
    }
    return analysis_results


def load_yolo_labels(annotations_path, class_names):
    """ Загрузка меток классов из YOLO аннотаций. """
    dict_labels = dict()
    labels = list()
    for filename in annotations_path:
        name_foler = list(Path(filename).parts)[-3]
        if filename.endswith('.txt'):
            with open(filename, 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        labels.append(class_names[class_id])
        dict_labels[name_foler] = labels
    return dict_labels, labels


def gini_coefficient(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_examples = len(labels)
    gini = 0
    for label in class_counts:
        label_prob = class_counts[label] / total_examples
        gini += label_prob * (1 - label_prob)
    return gini


def calculate_class_imbalance(labels):
    class_counts = Counter(labels)
    max_count = max(class_counts.values())
    average_count = sum(class_counts.values()) / len(class_counts)
    overall_imbalance = max_count / average_count
    return overall_imbalance


def get_image_size(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        return width, height
    return None


def analysis_stats(images_path, annotations_path, classes_path, run_path):
    class_names = load_class_names(classes_path)
    dict_labels, class_labels = load_yolo_labels(annotations_path, class_names)
    gini = "{:.2f}".format(gini_coefficient(class_labels))
    plot_class_balance(class_labels, run_path)
    dumpCSV(class_names, class_labels, dict_labels, run_path)
    imbalance_ratio = calculate_class_imbalance(class_labels)
    image_count = len(images_path)
    number_of_classes = len(set(class_labels))
    img_w, img_h = get_image_size(images_path[0])
    analysis_results = {
        'W': img_w,
        'H': img_h,
        'Class Imbalance Gini': gini,
        'Class Imbalance Ratio': imbalance_ratio,
        'Number of images': image_count,
        'Number of classes': number_of_classes,
    }
    return analysis_results