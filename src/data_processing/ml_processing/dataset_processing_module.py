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
from src.data_processing.ml_processing.annotation_analysis import analysis_yolo_annotations
from src.data_processing.ml_processing.image_analysis import analysis_stats, analysis_image_dataset


def find_paths(data_path, image_mode = True): #info_processor older find_image
    supported_extensions = {".jpg", ".jpeg", ".png"} if image_mode else {".txt"}
    paths = []
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        for root, dirs, files in os.walk(path / name):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_extensions:
                    paths.append(os.path.join(root, file))

    return paths


def feature_extraction(dataset_path, classes_path, run_path):
    images_path = find_paths(dataset_path, image_mode=True)
    annotations_path = find_paths(dataset_path, image_mode=False)

    analyze_image, analyze_color_stats = analysis_image_dataset(images_path)
    analyze_annotations = analysis_yolo_annotations(annotations_path)
    analyze_stat = analysis_stats(images_path, annotations_path, classes_path, run_path)

    df_analyze_color_stats = pd.DataFrame([analyze_image])
    df_color_stats = pd.DataFrame([pd.DataFrame(analyze_color_stats).mean().to_dict()])
    df_analyze_annotations = pd.DataFrame([analyze_annotations])
    df_analyze_stats = pd.DataFrame([analyze_stat])
    df_dataset_features = pd.concat([df_analyze_color_stats, df_color_stats, df_analyze_annotations, df_analyze_stats], axis=1)
    df_dataset_features.to_csv(run_path / 'dataset_features.csv', index=False)


    return df_dataset_features


    

