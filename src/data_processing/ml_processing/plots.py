from collections import Counter
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

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



def plot_with_lines_and_predictions(train_pca, test_pca, train_labels, names_test,  predicted_labels, names_train, ax, title, encoder):
    unique_labels = np.unique(train_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    legend_elements = []

    added_test_labels = set()

    for k, col in zip(unique_labels, colors):
        class_member_mask = (train_labels == k)
        xy = train_pca[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], edgecolor='black', alpha=0.75)
        if len(xy) > 1:
            center = np.mean(xy, axis=0)
            radius = np.max(np.linalg.norm(xy - center, axis=1))
            circle = plt.Circle(center, radius, color=col, fill=False, lw=2, linestyle='--')
            ax.add_patch(circle)

    for i, point in enumerate(test_pca):
        pred_label = predicted_labels[i]
        color = colors[unique_labels.tolist().index(pred_label)] if pred_label in unique_labels else 'gray'

        ax.scatter(point[0], point[1], s=100, c=[color], marker='*', edgecolor='black', alpha=0.75)


        bbox_edgecolor = 'black' 
        ax.text(point[0], point[1], names_test, fontsize=9, color='green', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=bbox_edgecolor, lw=1))

        train_idx = names_train[names_train == encoder.inverse_transform([pred_label.ravel()])[0]].index[0]


        train_point = train_pca[train_idx]
        ax.plot([point[0], train_point[0]], [point[1], train_point[1]], 'k--', linewidth=1)
        ax.text(train_point[0], train_point[1], names_train.iloc[train_idx], fontsize=9, color='black', ha='right', va='top')

    true_patch = mpatches.Patch(edgecolor='black', facecolor='white', label='True', lw=1)
    false_patch = mpatches.Patch(edgecolor='red', facecolor='white', label='False', lw=1)
    legend_elements.append(true_patch)
    legend_elements.append(false_patch)
    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    
