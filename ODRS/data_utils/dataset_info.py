import cv2
import os
import numpy as np


def get_count_classes(classes_path):
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as file:
            classes = [line.strip() for line in file.readlines()]

        count_classes = len(classes)
    else:
        print("File 'classes.txt' does not exist in the folder.")
    return count_classes


def gini_coefficient(labels):
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    total_examples = len(labels)
    gini = 0
    for label in class_counts:
        label_prob = class_counts[label] / total_examples
        gini += label_prob * (1 - label_prob)
    return gini


def process_txt_file(file_path, classes):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                class_label = line.split()[0]
                classes.append(class_label)
            except Exception:
                pass


def process_directory(directory_path, classes):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            if file_name == 'labels':
                label_dir = file_path
                for label_file_name in os.listdir(label_dir):
                    label_file_path = os.path.join(label_dir, label_file_name)
                    if label_file_name.endswith('.txt'):
                        process_txt_file(label_file_path, classes)
        elif file_name.endswith('.txt'):
            process_txt_file(file_path, classes)


def process_dataset(dataset_path):
    classes = []
    if os.path.exists(dataset_path):
        if os.path.isdir(dataset_path):
            for split in ['train', 'valid', 'test', 'val']:
                split_path = os.path.join(dataset_path, split)
                if os.path.exists(split_path):
                    process_directory(split_path, classes)
        else:
            process_txt_file(dataset_path, classes)
    return classes


def get_image_size(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        return width, height
    return None


def count_images_in_directory(directory_path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_count = 0
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in image_extensions):
            image_count += 1
    return image_count


def process_directory_img(directory_path):
    image_count = 0
    image_size = None

    label_path = os.path.join(directory_path, 'labels')
    image_path = os.path.join(directory_path, 'images')

    if os.path.isdir(label_path) and os.path.isdir(image_path):
        image_count = count_images_in_directory(image_path)
    else:
        image_count = count_images_in_directory(directory_path)

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            sub_image_count, sub_image_size = process_directory_img(file_path)
            image_count += sub_image_count
            if sub_image_size is not None:
                image_size = sub_image_size
        elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_size = get_image_size(file_path)

    return image_count, image_size


def process_dataset_img(dataset_path):
    image_count = 0
    image_size = None

    if os.path.isdir(dataset_path):
        for split in ['train', 'valid', 'test', 'val']:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                sub_image_count, sub_image_size = process_directory_img(split_path)
                image_count += sub_image_count
                if sub_image_size is not None:
                    image_size = sub_image_size
    else:
        image_count, image_size = process_directory_img(dataset_path)

    return image_count, image_size


def dataset_info(dataset_path, classes_path):
    class_labels = process_dataset(dataset_path)
    gini = "{:.2f}".format(gini_coefficient(class_labels))
    image_count, image_size = process_dataset_img(dataset_path)

    print("Number of images:", image_count)
    print("W:", image_size[0])
    print("H:", image_size[1])
    print(f"Gini Coefficient: {float(gini) * 100}")
    print("Number of classes:", get_count_classes(classes_path))

    return [float(image_size[0]), float(image_size[1]), float(gini) * 100,
            float(get_count_classes(classes_path)), float(image_count)]
