import cv2
import numpy as np
import random
from aug_image import noisy_salt_and_pepper_img, noisy_gauss_img, brightness_img
from aug_image import blur_img, saturation_img
from aug_image import flip_horizontol_img, flip_vertical_img, rotate_img
from PIL import Image

def get_boxes(img, annotation_path):
    with open(annotation_path, 'r') as annotation_file:
        lines = annotation_file.readlines()

    boxes = []
    for line in lines:
        data = line.strip().split()
        class_id, x_center, y_center, width, height = map(float, data)
        x_center *= img.shape[1]
        y_center *= img.shape[0]
        width *= img.shape[1]
        height *= img.shape[0]
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        boxes.append((x1, y1, x2, y2))
    return boxes


def flip_horizontol_within_box(image_path, annotation_path):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        flip_box_h = flip_horizontol_img(box_region)
        flip_box_h = cv2.cvtColor(np.array(flip_box_h), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = flip_box_h
    cv2.imwrite('image.jpg', image)


def flip_vertical_within_box(image_path, annotation_path):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        flip_box_w = flip_vertical_img(box_region)
        flip_box_w = cv2.cvtColor(np.array(flip_box_w), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = flip_box_w
    cv2.imwrite('image.jpg', image)
    

def rotate_within_box(image_path, annotation_path, value):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        rotate_box = rotate_img(box_region, value)
        rotate_box = cv2.cvtColor(np.array(rotate_box), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = rotate_box
    cv2.imwrite('image.jpg', image)


def noisy_salt_and_pepper_within_box(image_path, annotation_path,  noise_level):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        noisy_box = noisy_salt_and_pepper_img(box_region, noise_level)
        image[y1:y2, x1:x2] = noisy_box
    cv2.imwrite('image.jpg', image)


def noisy_gauss_within_box(image_path, annotation_path,  std):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        noisy_box = noisy_gauss_img(box_region, std)
        image[y1:y2, x1:x2] = noisy_box
    # cv2.imwrite('image.jpg', image)


def brightness_within_box(image_path, annotation_path, std):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        image_box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        brightness_box = brightness_img(image_box_region, std)
        brightness_box = cv2.cvtColor(np.array(brightness_box), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = brightness_box
    # cv2.imwrite('image.jpg', image)


def blur_within_box(image_path, annotation_path, blur_radius):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        image_box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        brightness_box = blur_img(image_box_region, blur_radius)
        brightness_box = cv2.cvtColor(np.array(brightness_box), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = brightness_box
    # cv2.imwrite('image.jpg', image)


def saturation_within_box(image_path, annotation_path, blur_radius):
    image = cv2.imread(image_path)
    boxes = get_boxes(image, annotation_path)
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_region = image[y1:y2, x1:x2]
        image_box_region = Image.fromarray(cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB))
        saturation_box = saturation_img(image_box_region, blur_radius)
        saturation_box = cv2.cvtColor(np.array(saturation_box), cv2.COLOR_RGB2BGR)
        image[y1:y2, x1:x2] = saturation_box
    # cv2.imwrite('image.jpg', image)







# image_path = '/media/space/ssd_1_tb_evo_sumsung/Work/Warp-D/test/images/Monitoring_photo_2_test_25-Mar_11-09-46.jpg'
# annotation_path = '/media/space/ssd_1_tb_evo_sumsung/Work/Warp-D/test/labels/Monitoring_photo_2_test_25-Mar_11-09-46.txt'
# # noisy_gauss_within_box(image_path, annotation_path, 17)
# # noisy_salt_and_pepper_within_box(image_path, annotation_path, 0.7)
# # brightness_within_box(image_path, annotation_path, 0.7)
# # blur_within_box(image_path, annotation_path, 1.5)
# # saturation_within_box(image_path, annotation_path, 3)

# # flip_horizontol_within_box(image_path, annotation_path)
# # flip_vertical_within_box(image_path, annotation_path)
# rotate_within_box(image_path, annotation_path, 75)
