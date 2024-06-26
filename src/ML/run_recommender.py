import sys
import os
import yaml
from pathlib import Path
from loguru import logger

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from src.data_processing.data_utils.utils import  create_run_directory, get_data_path, get_classes_path, load_config
from src.data_processing.data_utils.split_dataset import split_data
from src.data_processing.ml_processing.dataset_processing_module import feature_extraction
from src.data_processing.ml_processing.recommendation_module import predict_models


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # PATH TO ODRS
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_ml_config_data(path_config):
    config = load_config(path_config)
    data_config_ml = {
        'mode': config['GPU'],
        'classes_path': config['classes_path'],
        'dataset_path': config['dataset_path'],
        'speed': config['speed'],
        'accuracy': config['accuracy'],
        'balance': config['balance']
    }
    return data_config_ml

    
def dump_yaml(data_config_ml, top_models, run_path):
    data = {'GPU': data_config_ml['mode'],
            'accuracy': data_config_ml['accuracy'],
            'classes_path': str(data_config_ml['classes_path']),
            'dataset_path': str(data_config_ml['dataset_path']),
            'speed': data_config_ml['speed'],
            'balance':data_config_ml['balance'],
            'Top_1': top_models[0],
            'Top_2': top_models[1],
            'Top_3': top_models[2]
            }
    with open(run_path / 'results.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def predict(data_config_ml):
    file = Path(__file__).resolve()
    run_path = create_run_directory(model='ml')

    dataset_path = get_data_path(ROOT, data_config_ml['dataset_path'])
    data_config_ml['dataset_path'] = dataset_path
    classes_path = get_classes_path(ROOT, data_config_ml['classes_path'])
    data_config_ml['classes_path'] = classes_path
    split_data(dataset_path, split_train_value=0.75, split_valid_value=0.15)

    # get and save data features
    df_dataset_features = feature_extraction(dataset_path, classes_path, run_path)
    df_dataset_features.to_csv(run_path / 'dataset_features.csv', index=False)

    top_models = predict_models(df_dataset_features, data_config_ml, run_path)
    logger.info("Top models for training:")
    for i, model in enumerate(top_models):
        logger.info(f'{i + 1}) {model}')

    dump_yaml(data_config_ml, top_models, run_path)
    

def ml_main():
    file = Path(__file__).resolve()
    data_config_ml = get_ml_config_data(Path(file.parents[0]) / 'config' / 'ml_config.yaml')
    predict(data_config_ml)

if __name__ == "__main__":
    ml_main()
