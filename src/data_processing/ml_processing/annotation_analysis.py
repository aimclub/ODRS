import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os

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