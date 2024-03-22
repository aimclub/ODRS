import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path


def check_filename(filename):
    if '&' in filename:
        return False
    else:
        return True


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes, classes, difficulties = [], [], []
    for object in root.iter('object'):
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text) - 1
        ymin = int(bndbox.find('ymin').text) - 1
        xmax = int(bndbox.find('xmax').text) - 1
        ymax = int(bndbox.find('ymax').text) - 1
        boxes.append([xmin, ymin, xmax, ymax])

        label = object.find('name').text.lower().strip()
        classes.append(label)

        difficulty = int(object.find('difficult').text == '1')
        difficulties.append(difficulty)

    return boxes, classes, difficulties


def save_as_json(basename, dataset):
    filename = os.path.join(os.path.dirname(__file__), basename)
    print("Saving %s ..." % filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)


def read_names_from_txt(txt_path):
    names = []
    with open(txt_path, 'r') as file:
        for line in file:
            name = line.strip()
            if name:
                names.append(name)
    return names


def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            name = os.path.splitext(filename)[0]
            image_names.append(name)
    return image_names


def create_ssd_json(path_folder, txt_path):
    current_file_path = Path(__file__).resolve()
    txt_path = Path(current_file_path.parents[2]) / txt_path
    class_names = read_names_from_txt(txt_path)

    paths = {
        2007: os.path.join(os.path.dirname(path_folder), path_folder)
    }

    dataset = []
    for year, path in paths.items():
        ids = get_image_names(Path(path_folder) / 'images')
        for id in tqdm(ids):
            image_path = os.path.join(path, 'images', id + '.jpg')
            annotation_path = os.path.join(path, 'annotations', id + '.xml')
            if check_filename(annotation_path):
                try:
                    boxes, classes, difficulties = parse_annotation(annotation_path)
                    classes = [class_names.index(c) for c in classes]
                    dataset.append(
                        {
                            'image': os.path.abspath(image_path),
                            'boxes': boxes,
                            'classes': classes,
                            'difficulties': difficulties
                        }
                    )
                except Exception as e:
                    print(e)

        save_as_json(Path(os.path.dirname(path_folder)) / f'{path_folder.name}.json', dataset)
