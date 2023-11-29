import os
import json
import glob
from PIL import Image
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



def resize_images_and_annotations(data_path, img_size):
    size = img_size if img_size <= 300 else 300
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        folder_path = path / name
        images_path = os.path.join(folder_path, 'images')
        labels_path = os.path.join(folder_path, 'labels')

        for image_name in tqdm(os.listdir(images_path), desc=f'Resize {name} images'):
            image_path = os.path.join(images_path, image_name)
            label_path = os.path.join(labels_path, image_name.replace('.jpg', '.txt'))

            with Image.open(image_path) as img:
                original_width, original_height = img.size

                if original_width > size or original_height > size:
                    img = img.resize((size, size))

                    if os.path.exists(label_path):
                        with open(label_path, 'r') as file:
                            lines = file.readlines()

                        with open(label_path, 'w') as file:
                            for line in lines:
                                parts = line.split()
                                if len(parts) == 5:
                                    x_center = float(parts[1]) * original_width
                                    y_center = float(parts[2]) * original_height
                                    width = float(parts[3]) * original_width
                                    height = float(parts[4]) * original_height

                                    x_center *= size / original_width
                                    y_center *= size / original_height
                                    width *= size / original_width
                                    height *= size / original_height

                                    x_center /= size
                                    y_center /= size
                                    width /= size
                                    height /= size

                                    file.write(f"{parts[0]} {x_center} {y_center} {width} {height}\n")

                    img.save(image_path)

# resize_images_and_annotations('/media/space/ssd_1_tb_evo_sumsung/ITMO/ODRS/user_datasets/Warp-D_voc/test')