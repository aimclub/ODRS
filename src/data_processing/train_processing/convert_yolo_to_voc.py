import os
import re
from pathlib import Path
import shutil
from loguru import logger
from PIL import Image
from tqdm import tqdm
from prepare_ssd import create_ssd_json



def convert_voc(data_path, txt_path):
    logger.info("Creating VOC format for dataset")
    path = Path(data_path)
    folder_names = [folder.name for folder in path.iterdir() if folder.is_dir()]
    for name in folder_names:
        convert_yolo_to_voc(Path(data_path) / name, txt_path, 'annotations')
        shutil.rmtree(Path(data_path) / name / 'labels')
        create_ssd_json(Path(data_path) / name, txt_path)


def copy_files_to_jpeg_images_folder(data_path):
    jpeg_images_folder = os.path.join(data_path, 'images')
    for subfolder in ["labels"]:
        subfolder_path = os.path.join(data_path, subfolder)
        if os.path.exists(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, jpeg_images_folder)
    return jpeg_images_folder


def delete_txt_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)


def convert_yolo_to_voc(data_path, txt_path, folder_annotations):
    current_file_path = Path(__file__).resolve()
    jpeg_images_folder = copy_files_to_jpeg_images_folder(data_path)

    def is_number(n):
        try:
            float(n)
            return True
        except ValueError:
            return False

    folder_holding_yolo_files = jpeg_images_folder
    yolo_class_list_file = f"{current_file_path.parents[2]}/{txt_path}"

    # Get a list of all the classes used in the YOLO format
    with open(yolo_class_list_file) as f:
        yolo_classes = f.readlines()
    array_of_yolo_classes = [x.strip() for x in yolo_classes]

    os.chdir(folder_holding_yolo_files)

    if not os.path.exists(os.path.join(folder_holding_yolo_files, folder_annotations)):
        os.mkdir(folder_annotations)

    for each_yolo_file in tqdm(os.listdir(folder_holding_yolo_files)):
        if each_yolo_file.endswith("txt"):
            the_file = open(each_yolo_file, 'r')
            all_lines = the_file.readlines()
            image_name = each_yolo_file

            for ext in ['jpeg', 'jpg', 'png']:
                image_path = os.path.join(folder_holding_yolo_files, each_yolo_file.replace('txt', ext))
                if os.path.exists(image_path):
                    image_name = each_yolo_file.replace('txt', ext)
                    break

            if image_name == each_yolo_file:
                continue

            orig_img = Image.open(image_path)  # Open the image
            image_width = orig_img.width
            image_height = orig_img.height

            with open(os.path.join(folder_annotations, each_yolo_file.replace('txt', 'xml')), 'w') as f:
                f.write('<annotation>\n')
                f.write(f'\t<folder>{folder_annotations}</folder>\n')
                f.write('\t<filename>' + image_name + '</filename>\n')
#                f.write('\t<path>' + os.path.join(os.getcwd(), image_name) + '</path>\n')
                f.write('\t<source>\n')
                f.write('\t\t<database>Unknown</database>\n')
                f.write('\t</source>\n')
                f.write('\t<size>\n')
                f.write('\t\t<width>' + str(image_width) + '</width>\n')
                f.write('\t\t<height>' + str(image_height) + '</height>\n')
                f.write('\t\t<depth>3</depth>\n')  # Assuming a 3-channel color image (RGB)
                f.write('\t</size>\n')
                f.write('\t<segmented>0</segmented>\n')

                for each_line in all_lines:
                    yolo_array = re.split(r'\s', each_line.rstrip())  # Remove any extra space from the end of the line

                    class_number = 0
                    x_yolo = 0.0
                    y_yolo = 0.0
                    yolo_width = 0.0
                    yolo_height = 0.0
                    yolo_array_contains_only_digits = True

                    if len(yolo_array) == 5:
                        for each_value in yolo_array:
                            if not is_number(each_value):
                                yolo_array_contains_only_digits = False

                        if yolo_array_contains_only_digits:
                            class_number = int(yolo_array[0])
                            object_name = array_of_yolo_classes[class_number]
                            x_yolo = float(yolo_array[1])
                            y_yolo = float(yolo_array[2])
                            yolo_width = float(yolo_array[3])
                            yolo_height = float(yolo_array[4])

                            box_width = yolo_width * image_width
                            box_height = yolo_height * image_height
                            x_min = int(x_yolo * image_width - (box_width / 2))
                            y_min = int(y_yolo * image_height - (box_height / 2))
                            x_max = int(x_yolo * image_width + (box_width / 2))
                            y_max = int(y_yolo * image_height + (box_height / 2))

                            if x_min <= 0:
                                x_min = 3
                            if x_max > image_width:
                                x_max = image_width

                            if y_min <= 0:
                                y_min = 3
                            if y_max > image_height:
                                y_max = image_height

                            x_min = str(x_min)
                            y_min = str(y_min)
                            x_max = str(x_max)
                            y_max = str(y_max)
                            f.write('\t<object>\n')
                            f.write('\t\t<name>' + object_name + '</name>\n')
                            f.write('\t\t<pose>Unspecified</pose>\n')
                            f.write('\t\t<truncated>0</truncated>\n')
                            f.write('\t\t<difficult>0</difficult>\n')
                            f.write('\t\t<bndbox>\n')
                            f.write('\t\t\t<xmin>' + x_min + '</xmin>\n')
                            f.write('\t\t\t<xmax>' + x_max + '</xmax>\n')
                            f.write('\t\t\t<ymin>' + y_min + '</ymin>\n')
                            f.write('\t\t\t<ymax>' + y_max + '</ymax>\n')
                            f.write('\t\t</bndbox>\n')
                            f.write('\t</object>\n')

                f.write('</annotation>\n')

    if os.path.exists(os.path.join(folder_holding_yolo_files, folder_annotations)):
        print("Conversion complete")
    else:
        print("There was a problem converting the files")

    shutil.move(f"{folder_holding_yolo_files}/{folder_annotations}", data_path)
    delete_txt_files_in_folder(folder_holding_yolo_files)
