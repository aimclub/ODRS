import os
import sys
import pandas as pd
import catboost as cat
from pathlib import Path
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
file = Path(__file__).resolve()
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))

from src.data_processing.ml_processing.plots import plot_with_lines_and_predictions


def ml_predict(df_rules, df_dataset_features, run_path):
    scaler = StandardScaler()
    encoder = LabelEncoder()
    mds = umap.UMAP()

    cols_to_drop = [col for col in df_rules.columns if col.startswith(('Min', 'Max'))]
    df_rules = df_rules.drop(columns=cols_to_drop)

    cols_to_drop = [col for col in df_dataset_features.columns if col.startswith(('Min', 'Max'))]
    df_dataset_features = df_dataset_features.drop(columns=cols_to_drop)

    X_train = df_rules.drop('Dataset', axis=1)
    y_train = df_rules['Dataset']

    X_test = df_dataset_features

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = encoder.fit_transform(y_train.values.ravel())

    X_umap = mds.fit_transform(X_train) # for plot
    X_umap_test = mds.transform(X_test) # for plot
    train_dataset_names = df_rules['Dataset']

    model = cat.CatBoostClassifier(iterations=100, learning_rate=0.1, random_strength=6, verbose=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    plot_with_lines_and_predictions(X_umap, X_umap_test, y_train, "Current Dataset", y_pred, train_dataset_names, ax, 'ML Predictions', encoder)
    plt.savefig(run_path / "Prediction_ml.png")
    return encoder.inverse_transform(y_pred.ravel())[0]


def evaluate_items(data, mode=False, accuracy=False, speed=False, balance=False):
    if (speed and accuracy) or (speed and balance) or (accuracy and balance):
        raise Exception("Select only one of the Balance, Speed, or Accuracy options.")

    fps = "FPS_CPU" if not mode else "FPS_GPU"
    data = data.drop("FPS_CPU" if mode else "FPS_GPU", axis=1)
    mAP = "mAP50"

    # Убедитесь, что данные в mAP50 числовые
    data[mAP] = pd.to_numeric(data[mAP], errors='coerce')
    data[fps] = pd.to_numeric(data[fps], errors='coerce')

    if accuracy:
        top_three_indices = data[mAP].nlargest(3).index
        top_models = data.loc[top_three_indices, 'Model']
    elif speed:
        top_three_indices = data[fps].nlargest(3).index
        top_models = data.loc[top_three_indices, 'Model']
    elif balance:
        # Нормализация FPS
        fps_percentage = data[fps] / data[fps].max() * 100
        data[f'{fps}_percent'] = fps_percentage

        # Нормализация mAP50
        mAP_percentage = data[mAP] / data[mAP].max() * 100
        data[f'{mAP}_percent'] = mAP_percentage

        # Создание совокупного балансового показателя
        data['Balance_Score'] = data[f'{fps}_percent'] + data[f'{mAP}_percent']

        # Получение топ-3 моделей по балансовому показателю
        top_three_indices = data['Balance_Score'].nlargest(3).index
        top_models = data.loc[top_three_indices, 'Model']
    else:
        raise Exception("Select only one of the Balance, Speed, or Accuracy options.")
    return top_models



def predict_models(df_dataset_features, data_config_ml, run_path):
    df_rules = pd.read_csv(Path(file.parents[1]) / 'data_rules' / 'datasets.csv', delimiter=';')
    df_metrics = pd.read_csv(Path(file.parents[1]) / 'data_rules' / 'rules.csv', delimiter=';')
    dataset_name = ml_predict(df_rules, df_dataset_features, run_path)
    pop_rows = df_metrics.loc[df_metrics['Dataset'] == dataset_name]
    pop_rows = pop_rows.drop(columns=['P', 'R', 'Dataset', 'mAP95'])
    models = evaluate_items(pop_rows, data_config_ml['mode'], data_config_ml['accuracy'], data_config_ml['speed'], data_config_ml['balance'])
    return models.tolist()



