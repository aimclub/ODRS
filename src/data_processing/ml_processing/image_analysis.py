from collections import Counter
from tqdm import tqdm
import cv2
from PIL import Image, ImageStat
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def calculate_exposure_wbalance(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Преобразование в RGB (OpenCV загружает в формате BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Анализ баланса белого по средним значениям RGB
    average_color_per_row = np.average(image_rgb, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    average_color = np.uint8(average_color)
    # print(f"Estimated White Balance (Average RGB): {average_color.mean()}")

    # Преобразование в градации серого и расчет гистограммы
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()

    # Вычисление метрики Exposure Value
    exposure_value = -np.sum(histogram * np.log2(histogram + 1e-5))
    # print(f"Estimated Exposure Value: {exposure_value:.2f}")

    return average_color.mean(), exposure_value

def calculate_CNN_stats(image_path):
    # Определение класса модели
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            # Свёрточный слой (принимает 3 канала, выходит 16 каналов, ядро размером 3x3)
            self.conv_layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            # Слой глобального усредняющего пулинга
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            # Применение свёрточного слоя
            x = self.conv_layer(x)
            # Применение нелинейности (ReLU)
            x = F.relu(x)
            # Применение глобального усредняющего пулинга
            x = self.global_avg_pool(x)
            # Сжатие данных до одного значения путем усреднения всех каналов
            x = torch.flatten(x, 1)  # Преобразуем в плоский вид
            x = x.mean(dim=1)        # Усреднение по всем каналам
            return x
        
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Изменение размера изображения
        transforms.ToTensor()           # Преобразование изображения в тензор PyTorch
    ])
    image = transform(image).unsqueeze(0)
    model = ConvNet()

    # Получение Scalar averange pooling
    try:
        scalar_averange_pooling = model(image)
    except:
        scalar_averange_pooling = 0

    image = image.float()  # Убедимся, что тип данных float32

    # Определение фильтров для границ, углов и текстур
    sobel_filter = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0)
    edge_filter = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0)
    texture_filter = torch.tensor([[0, 1, 0], [-1, -4, -1], [0, 1, 0]], dtype=torch.float32).repeat(3, 1, 1).unsqueeze(0)

    # Создаем сверточные слои
    conv_sobel = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
    conv_edge = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
    conv_texture = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)

    # Задаем веса сверточных слоев
    conv_sobel.weight.data = sobel_filter
    conv_edge.weight.data = edge_filter
    conv_texture.weight.data = texture_filter

    # Применение фильтров
    sobel_output = conv_sobel(image)
    edge_output = conv_edge(image)
    texture_output = conv_texture(image)

    # Усреднение результатов
    mean_sobel = sobel_output.mean()
    mean_edge = edge_output.mean()
    mean_texture = texture_output.mean()

    # Вывод усреднённых значений для каждого фильтра

    return scalar_averange_pooling.item(),mean_sobel.item(), mean_edge.item(), mean_texture.item()


def calculate_color_stats_and_histograms(image_path):
    img_rgb = cv2.imread(image_path)
    img_rgb_32F = np.float32(img_rgb)
    img_hsv_32F = cv2.cvtColor(img_rgb_32F, cv2.COLOR_BGR2HSV)
    B, G, R = cv2.split(img_rgb)
    H_32, S_32, V_32 = cv2.split(img_hsv_32F)

    H_32 = H_32 / 360.0
    V_32 = V_32 / 255.0

    def compute_hist_and_stats(channel, bins=256, range=(0, 256), is_normalized=True):
        hist, _ = np.histogram(channel.ravel(), bins, range)
        if is_normalized:
            hist = hist.astype(float) / sum(hist)
        mean = stats.tmean(channel.ravel())
        std = stats.tstd(channel.ravel())
        return hist, mean, std

    stats_dict = {}
    channels = [B, G, R, H_32, S_32, V_32]
    channel_names = ['B', 'G', 'R', 'H', 'S', 'V']
    ranges = [(0, 256), (0, 256), (0, 256), (0, 1), (0, 1), (0, 256)]
    normalize_flags = [True, True, True, True, True, True]

    for name, channel, rng, norm_flag in zip(channel_names, channels, ranges, normalize_flags):
        hist, mean, std = compute_hist_and_stats(channel, range=rng, is_normalized=norm_flag)
        stats_dict[f'{name}_hist'] = np.std(hist)
        stats_dict[f'{name}_mean'] = mean
        stats_dict[f'{name}_std'] = std


    return stats_dict


def analysis_image_dataset(images_path):
    analyze_color_stats = []
    diversity_list = []
    brightness_list = []
    contrast_list = []
    entropy_list = []
    #for CNN
    sap = []
    ms = []
    mt = []
    me = []

    #for calculate_exposure_wbalance
    ac = []
    ev = []

    for image_path in tqdm(images_path, desc="Image analyze"):
        analyze_color_stats.append(calculate_color_stats_and_histograms(image_path))
        try:
            scalar_averange_pooling, mean_sobel, mean_edge, mean_texture = calculate_CNN_stats(image_path)
        except:
            continue
        average_color, exposure_value = calculate_exposure_wbalance(image_path)

        sap.append(scalar_averange_pooling)
        ms.append(mean_sobel)
        me.append(mean_edge)
        mt.append(mean_texture)
        ac.append(average_color)
        ev.append(exposure_value)

        with Image.open(image_path) as img:
            # Разнообразие фона
            colors = img.convert("RGB").getcolors(maxcolors=10000)
            diversity = len(colors) if colors is not None else 10000
            diversity_list.append(diversity)
            # Энтропия
            entropy = img.convert("RGB").entropy()
            entropy_list.append(entropy)
            # Освещенность и контраст
            grey_img = img.convert("L")
            stat = ImageStat.Stat(grey_img)
            brightness = stat.mean[0]
            contrast = stat.stddev[0]
            brightness_list.append(brightness)
            contrast_list.append(contrast)
    
    analyze_image = {
        'Average Diversity': np.mean(diversity_list),
        'Average Brightness': np.mean(brightness_list),
        'Average Contrast': np.mean(contrast_list),
        'Average Entropy': np.mean(entropy_list),
        'Max Diversity': np.max(diversity_list),
        'Min Diversity': np.min(diversity_list),
        'Max Brightness': np.max(brightness_list),
        'Min Brightness': np.min(brightness_list),
        'Max Contrast': np.max(contrast_list),
        'Min Contrast': np.min(contrast_list),
        'Max Entropy': np.max(entropy_list),
        'Min Entropy': np.min(entropy_list),
        'Scalar Averange Pooling': np.mean(sap),
        'Average sobel': np.mean(ms),
        'Average edge': np.mean(me),
        'Average texture': np.mean(mt),
        'Averange Color': np.mean(ac),
        'Averange exposure value': np.mean(ev)

    } 

    return analyze_image, analyze_color_stats