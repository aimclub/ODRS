from collections import Counter
import matplotlib.pyplot as plt
import random
from pathlib import Path


def plot_class_balance(labels, output_path):
    """ Построение и сохранение графика баланса классов с наклоненными метками и вывод среднего значения. """
    class_counts = Counter(labels)
    output_file = Path(output_path) / 'Classes_balance.png'
    colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in class_counts.keys()]

    plt.bar(class_counts.keys(), class_counts.values(), color=colors)
    plt.xlabel('Classes')
    plt.ylabel('Number of instances')
    plt.title('Class balance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
