from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def flip_horizontol_img(img):
    img_horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img_horizontal_flip


def flip_vertical_img(img):
    img_vertical_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img_vertical_flip


def rotate_img(img, value, expand=False):
    img_rotated = img.rotate(value, expand=expand)
    return img_rotated


def brightness_img(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img_brightened = enhancer.enhance(brightness_factor)
    return img_brightened


def noisy_gauss_img(img, std):
    np_img = np.array(img)
    gauss = np.random.normal(0, std, np_img.shape)
    noisy_gauss_img = np_img + gauss
    noisy_gauss_img = np.clip(noisy_gauss_img, 0, 255).astype(np.uint8)
    noisy_gauss_img = Image.fromarray(noisy_gauss_img)
    return noisy_gauss_img


def noisy_salt_and_pepper_img(img, noise_level):
    np_img = np.array(img)
    amount = int(np.ceil(noise_level * np_img.size / 3))
    coords = [np.random.randint(0, i - 1, amount) for i in np_img.shape]
    np_img[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, amount) for i in np_img.shape]
    np_img[coords[0], coords[1]] = 0
    noisy_salt_pepper_img = Image.fromarray(np_img)
    return noisy_salt_pepper_img


def blur_img(img, blur_radius):
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def saturation_img(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    image_saturation = enhancer.enhance(saturation_factor)
    return image_saturation




img = Image.open('/media/space/ssd_1_tb_evo_sumsung/exp_pad/dataset/train_dataset/moja/train/images/16-Aug_12-29-54.png')



# Adjust the saturation level (1.0 is the original saturation)
# saturation_factor = 10  # You can change this value to adjust saturation (e.g., 0.5 for less saturation)
# image_saturation = saturation_img(img, saturation_factor)
# image_saturation.save('modified_image.jpg')

# blur_radius = 1.5  # You can change this value to adjust the blur level
# blurred_image = blur_img(img, blur_radius)

# # Save the blurred image
# blurred_image.save('blurred_image.png')


# noisy_salt_pepper_img = noisy_salt_and_pepper(img, 0.2)
# noisy_salt_pepper_img.save('noisy_salt_pepper_img.png')


# img_noisy_gauss = noisy_gauss_img(img, 40)
# img_noisy_gauss.save('noisy_gauss.png')


# img_brightness = brightness_img(img, 1.5)
# img_brightness.save('brightness.png')

# img_horizontal_flip = flip_horizontol(img)
# img_horizontal_flip.save('horizontal_flip.png')

# img_vertical_flip = flip_vertical(img)
# img_vertical_flip.save('img_vertical_flip.png')

# # img_diagonal_flip_main = img.transpose(Image.Transpose.ROTATE_180)  # Поворот на 90 градусов
# # img_diagonal_flip_main.save('diagonal_flip_main.png')


# img_rotated = rotate(img, 75, expand=False)
# img_rotated.save('rotated_image.png')